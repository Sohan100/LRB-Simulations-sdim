#!/usr/bin/env python3
"""Adversarial critic of A4-nonmarkov-theory.md (C26 and the twirl
arguments of sections 2.2-2.4): is the repository's 216-string Clifford
catalog, as it acts on the logical qutrit of the [[5,1,2]]_3 code, the full
single-qutrit Clifford group (a unitary 2-design)?

Method (independent of A4's scripts):
  * physical gates from sdim.unitary (H = Fourier, P = diag(1,1,w), X, Z,
    CNOT|a,b> = |a,a+b>), CZ|a,b> = w^{ab}|a,b>, SWAP, *_INV = adjoint;
  * the repository's own logical-gate sequences
    (FoldedQutritDetectionCode.elementary_single_qudit_logical_operator_gate_sequence),
    encoding circuit (logical_plus_initial_state) and check circuits;
  * consistency checks of the conventions: the encoded |+>_L is stabilised
    by the brief's four stabilisers and by X_L = X0 X3; the Z- and X-check
    circuits give outcome 0 deterministically on code states; every logical
    token maps the code space to itself;
  * logical 3x3 matrix of every token, then of every catalog string
    (rightmost symbol acts first, QutritCliffordLibrary.qutrit_gate_seq_list);
  * number of distinct logical Cliffords mod phase, closure, frame potential
    of the uniform distribution over the 216 strings, twirl of a noise map.

    module load python/3.11
    python critic_A4-nonmarkov-theory_catalog.py          (~20 s)
"""
from __future__ import annotations

import itertools
import os
import sys
from collections import Counter

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WORKTREE = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
sys.path.insert(0, os.path.join(WORKTREE, "src"))
from sdim import unitary as su  # noqa: E402
from lrb.code_definitions import (  # noqa: E402
    FoldedQutritDetectionCode as Code,
    QutritCliffordLibrary as Lib,
    single_qutrit_cliffords_exhaustive_strings as STRINGS,
)

d = 3
w = np.exp(2j * np.pi / 3)
X1, Z1 = su.generate_x_matrix(d), su.generate_z_matrix(d)
H1, P1 = su.generate_h_matrix(d), su.generate_p_matrix(d)
CNOT = su.generate_cnot_matrix(d)
CZ = np.diag([w ** (a * b) for a in range(d) for b in range(d)])
SWAP = np.zeros((9, 9))
for a in range(3):
    for b in range(3):
        SWAP[3 * b + a, 3 * a + b] = 1
ONE = {"I": np.eye(3), "X": X1, "Z": Z1, "H": H1, "P": P1}
TWO = {"CNOT": CNOT, "CZ": CZ, "SWAP": SWAP}


def one_mat(name):
    if name.endswith("_INV"):
        return ONE[name[:-4]].conj().T
    return ONE[name]


def two_mat(name):
    if name.endswith("_INV"):
        return TWO[name[:-4]].conj().T
    return TWO[name]


def embed1(U, wire, n):
    ops = [np.eye(3)] * n
    ops[wire] = U
    out = ops[0]
    for o in ops[1:]:
        out = np.kron(out, o)
    return out


def embed2(U, a, b, n):
    """Two-qudit gate U (index 3*x_a + x_b) on wires a, b of n qutrits
    (wire 0 most significant)."""
    D = 3 ** n
    out = np.zeros((D, D), dtype=complex)
    digits = np.array(list(itertools.product(range(3), repeat=n)))
    U4 = U.reshape(3, 3, 3, 3)
    for col, dg in enumerate(digits):
        xa, xb = dg[a], dg[b]
        for ya in range(3):
            for yb in range(3):
                amp = U4[ya, yb, xa, xb]
                if amp != 0:
                    nd = dg.copy()
                    nd[a], nd[b] = ya, yb
                    row = int(np.dot(nd, 3 ** np.arange(n - 1, -1, -1)))
                    out[row, col] += amp
    return out


def circuit_unitary(circ, n):
    U = np.eye(3 ** n, dtype=complex)
    for op in circ.operations:
        name = op.gate_name
        if op.target_index is None:
            G = embed1(one_mat(name), op.qudit_index, n)
        else:
            G = embed2(two_mat(name), op.qudit_index, op.target_index, n)
        U = G @ U
    return U


def pauli(xs, zs, n):
    out = np.eye(1)
    for x, z in zip(xs, zs):
        out = np.kron(out, np.linalg.matrix_power(X1, x % 3) @
                      np.linalg.matrix_power(Z1, z % 3))
    return out


def key(U):
    f = U.ravel()
    k = int(np.flatnonzero(np.abs(f) > 1e-9)[0])
    return tuple(np.round((U * np.conj(f[k]) / abs(f[k])).ravel(), 6))


def main():
    n = 5
    S = {  # (x exponents, z exponents) on wires 0..4, as in the brief
        "S_Z1": ((0, 0, 0, 0, 0), (1, -1, 0, -1, 0)),
        "S_Z2": ((0, 0, 0, 0, 0), (0, 1, 1, 0, -1)),
        "S_X1": ((1, 1, -1, 0, 0), (0, 0, 0, 0, 0)),
        "S_X2": ((0, -1, 0, 1, -1), (0, 0, 0, 0, 0)),
    }
    Smat = {k: pauli(*v, n) for k, v in S.items()}
    XL = pauli((1, 0, 0, 1, 0), (0,) * 5, n)
    ZL = pauli((0,) * 5, (1, 0, 1, 0, 0), n)
    enc_circ = Code.logical_plus_initial_state()
    assert all(o.qudit_index < 5 and (o.target_index is None or
                                      o.target_index < 5)
               for o in enc_circ.operations)
    psi = circuit_unitary(enc_circ, 5)[:, 0]     # encoding of |00000>
    print("encoded |+>_L: <S> =", {k: np.round(np.vdot(psi, M @ psi), 6)
                                   for k, M in Smat.items()},
          " <X_L> =", np.round(np.vdot(psi, XL @ psi), 6))
    # projector onto the code space
    Pc = np.eye(3 ** n, dtype=complex)
    for M in Smat.values():
        Pc = Pc @ (np.eye(3 ** n) + M + M @ M) / 3
    print("code space dimension:", round(np.trace(Pc).real, 6))
    # logical basis
    zero = (np.eye(3 ** n) + ZL + ZL @ ZL) @ psi / 3
    zero /= np.linalg.norm(zero)
    basis = np.stack([np.linalg.matrix_power(XL, j) @ zero for j in range(3)],
                     1)
    print("Z_L on |j_L>: eigenvalues",
          np.round([np.vdot(basis[:, j], ZL @ basis[:, j]) for j in range(3)],
                   6))
    # check circuits: deterministic outcome 0 on code states
    for lab, fac, anc in (("Z", Code.z_ALT_stabilizer_measurement_ancillae_circuit, None),
                          ("X", Code.x_stabilizer_measurement_ancillae_circuit, None)):
        circ = fac()
        ops = [o for o in circ.operations if o.gate_name not in ("M", "M_X", "RESET")]
        wires = sorted({o.qudit_index for o in ops} |
                       {o.target_index for o in ops if o.target_index is not None})
        ancs = [q for q in wires if q >= 5]
        # simulate on data + these ancillas only
        m = {q: i for i, q in enumerate(list(range(5)) + ancs)}
        nn = 5 + len(ancs)
        Gs = []
        for o in ops:
            if o.target_index is None:
                Gs.append(embed1(one_mat(o.gate_name), m[o.qudit_index], nn))
            else:
                Gs.append(embed2(two_mat(o.gate_name), m[o.qudit_index],
                                 m[o.target_index], nn))
        for j in range(3):
            st = np.kron(basis[:, j], np.eye(3 ** len(ancs))[:, 0])
            for G in Gs:
                st = G @ st
            out = st.reshape(3 ** 5, *([3] * len(ancs)))
            probs = [np.sum(np.abs(out) ** 2, axis=tuple(
                i for i in range(len(ancs) + 1) if i != k + 1))
                for k in range(len(ancs))]
            print(f"  {lab}-check circuit on |{j}_L>: ancilla {ancs} outcome "
                  f"distributions {np.round(probs, 6).tolist()}")
    # logical tokens
    L = {}
    for tok in ("I", "X", "X_INV", "Z", "Z_INV", "H", "H_INV", "P", "P_INV"):
        U = circuit_unitary(Code.single_qudit_logical_gate_circuit(tok), n)
        leak = np.linalg.norm(Pc @ U @ Pc - U @ Pc)
        M = basis.conj().T @ U @ basis
        L[tok] = M
        print(f"  token {tok:5s}: code space preserved (|PUP-UP| = "
              f"{leak:.1e}); logical matrix unitary: "
              f"{np.allclose(M @ M.conj().T, np.eye(3))}; equals sdim "
              f"single-qutrit {tok} mod phase: "
              f"{key(M) == key(one_mat(tok))}")
    # catalog
    mats = []
    for s in STRINGS:
        U = np.eye(3, dtype=complex)
        for tok in Lib.qutrit_gate_seq_list(s):   # execution order
            U = L[tok] @ U
        mats.append(U)
    keys = [key(U) for U in mats]
    cnt = Counter(keys)
    kset = set(keys)
    closed = all(key(A @ B) in kset for A in mats for B in mats)
    fp2 = np.mean([[abs(np.trace(A.conj().T @ B)) ** 4 for B in mats]
                   for A in mats])
    print(f"\n  catalog: {len(STRINGS)} strings -> {len(kset)} distinct "
          f"logical Cliffords mod phase; closed: {closed}; frame potential "
          f"(uniform over strings) = {fp2:.6f} (2-design iff 2)")
    dups = [(k, c) for k, c in cnt.items() if c > 1]
    print(f"  elements appearing more than once: {len(dups)} (multiplicities "
          f"{sorted(c for _, c in dups)})")
    for k, c in dups:
        ss = [STRINGS[i] for i in range(len(STRINGS)) if keys[i] == k]
        print(f"     {c} strings for one element: {ss}")
    # the full group generated by the logical H and P, for comparison
    grp = {key(np.eye(3)): np.eye(3, dtype=complex)}
    front = [np.eye(3, dtype=complex)]
    while front:
        new = []
        for U in front:
            for G in (L["H"], L["P"]):
                V = G @ U
                kk = key(V)
                if kk not in grp:
                    grp[kk] = V
                    new.append(V)
        front = new
    missing = [V for kk, V in grp.items() if kk not in kset]
    print(f"  group generated by logical H and P: {len(grp)} elements; "
          f"missing from the catalog: {len(missing)}")
    # effect on the twirl: twirl a non-depolarising Pauli channel
    def sup(U):
        return np.kron(U, U.conj())
    Wey = [np.linalg.matrix_power(X1, a) @ np.linalg.matrix_power(Z1, b)
           for a in range(3) for b in range(3)]
    rng = np.random.default_rng(5)
    probs = rng.dirichlet(np.ones(8)) * 0.05
    E = (1 - probs.sum()) * np.eye(9, dtype=complex)
    for pr, W in zip(probs, Wey[1:]):
        E = E + pr * sup(W)
    tw = sum(sup(U).conj().T @ E @ sup(U) for U in mats) / len(mats)
    Id = np.eye(9)
    Dep = np.outer(np.eye(3).ravel() / 3, np.eye(3).ravel())
    A = np.stack([Id.ravel(), Dep.ravel()], 1)
    c, *_ = np.linalg.lstsq(A, tw.ravel(), rcond=None)
    ev = np.sort(np.linalg.eigvals(tw).real)
    print(f"  twirl over the catalog of a random Pauli channel (total error "
          f"0.05): residual from a Id + b Dep = "
          f"{np.abs(A @ c - tw.ravel()).max():.1e}; eigenvalues of the "
          f"twirled channel {np.round(ev, 6)} (a 2-design gives one "
          f"eigenvalue 1 and an eightfold degenerate one)")


def provenance():
    """Decode the forward Clifford strings of the baseline run's m = 22
    circuits (30 sequences) from their physical gate patterns and test which
    catalog they come from: the committed (defective) one in this worktree or
    the corrected one in the main checkout (9 Pauli words x 24 symplectic
    words, uncommitted, modified 2026-10-28)."""
    from sdim.circuit_io import read_circuit
    run = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-"
           "slurm/Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
    old = set(STRINGS)
    # corrected catalog, rebuilt from the main checkout's definition
    pw = ("", "X", "XX", "Z", "XZ", "XXZ", "ZZ", "XZZ", "XXZZ")
    import ast
    src = open("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/src/lrb/"
               "code_definitions.py").read()
    start = src.index("QUTRIT_SYMPLECTIC_WORDS = (")
    end = src.index(")", start)
    words = ast.literal_eval(src[start + len("QUTRIT_SYMPLECTIC_WORDS = "):
                                 end + 1])
    new = {(p + w if w != "I" else (p or "I")).replace("S", "P")
           for p in pw for w in words}
    print(f"\n  provenance: corrected catalog has {len(new)} distinct strings"
          f" ({len(words)} symplectic words)")
    pat = {("X", 0): "X", ("Z", 0): "Z", ("SWAP", 2): "H", ("P", 0): "P",
           ("I", 0): "I"}
    n_old = n_new = n_both = n_none = 0
    only_old, only_new = Counter(), Counter()
    for seq in range(30):
        circ = read_circuit(f"{run}/experiments/LRB/{seq}/4/8.chp")
        ops = [o for o in circ.operations if o.gate_name not in ("N1", "N2")
               and isinstance(o.qudit_index, (int, np.integer))]
        # split at check rounds (first gate of a round: CNOT 0 -> 5)
        seg, segs = [], []
        for o in ops:
            if o.gate_name == "CNOT" and o.qudit_index == 0 and \
                    o.target_index == 5:
                segs.append(seg)
                seg = []
            elif all(q < 5 for q in (o.qudit_index,
                                     o.target_index if o.target_index
                                     is not None else 0)):
                seg.append((o.gate_name, o.qudit_index))
        forward = segs[:22]
        forward[0] = forward[0][7:]                  # drop the encoding
        for s in forward:
            toks = [pat[g] for g in s if g in pat]
            string = "".join(reversed(toks))         # rightmost acts first
            a, b = string in old, string in new
            n_old += a and not b
            n_new += b and not a
            n_both += a and b
            n_none += not (a or b)
            if a and not b:
                only_old[string] += 1
            if b and not a:
                only_new[string] += 1
    print(f"  baseline run, 30 x 22 forward Cliffords: only in committed "
          f"(defective) catalog {n_old}, only in corrected catalog {n_new}, "
          f"in both {n_both}, in neither {n_none}")
    print(f"  examples only-old: {only_old.most_common(6)}; only-new: "
          f"{only_new.most_common(6)}")


if __name__ == "__main__":
    main()
    provenance()
