#!/usr/bin/env python3
"""A1: what correction can buy for the [[5,1,2]]_3 folded qutrit surface code.

Independent implementation (numpy + Python standard library only for the
main analysis).  Sections:

  A. Z_3 Pauli algebra, code definition, commutation checks, encoder check.
  B. Syndrome classification of all 81 syndromes from weight-0/1/2 errors,
     grouped into the 9 logical cosets X_L^a Z_L^b.            (task a)
  C. Exact code-capacity analysis by enumeration of all 9^5 error
     configurations: detection, full MLD, hybrid, posterior threshold.
     Results are exact integer polynomials in u = q/D.          (task b)
  D. Biased and qutrit-restricted noise.                       (task c)
  E. Weight-2 logical operators / minimum-weight coset representatives.
                                                                (task d)
  F. Logical gates H_L, P_L: symplectic action, syndrome map, logical map,
     effect on ambiguity (true prior vs naive prior).           (task f)
  P. Phenomenological model with measurement errors, leading-order
     single- and two-fault enumeration over repeated rounds.    (task e)
  G. Circuit level: own SI1000 cycle generator and Pauli-frame propagator,
     single-fault counting per check round, and leading-order analysis of
     baseline LRB circuits read from the (read-only) run folder.
     Optional --dem-check compares the propagator with the repository's
     CompactDetectorErrorModel (needs sdim + repo src on sys.path).

Conventions (all mod 3):
  * A Pauli on n qutrits is v = (x_0..x_{n-1}, z_0..z_{n-1}), meaning
    X^x Z^z up to a phase.
  * Symplectic form <A,B> = z_A.x_B - x_A.z_B, so that A B = w^<A,B> B A.
  * Syndrome digit of check S for data error E: sigma_S(E) = <S, E>.
    This equals the shift of the check's ancilla outcome in the repository
    circuits (checked in section G by propagation through the check block).
  * Logical label of E: a(E) = <Z_L,E> = x0 + x2, b(E) = -<X_L,E> = z0 + z3
    with X_L = X0 X3, Z_L = Z0 Z2.  For E in the normalizer,
    E ~ X_L^a Z_L^b * (stabilizer).  b(E) is exactly the shift of the
    terminal observable x0 + x3 (H_INV + M on wires 0 and 3).
  * Single-qutrit Pauli index k = x + 3 z (k = 0 is the identity).

Run (login node, one or two cores; see the log for the wall time):
    module load python/3.11
    OMP_NUM_THREADS=2 python A1_syndrome_classification.py [--dem-check]
Writes A1_results.json next to this script (or into --out).  Reads the
baseline run folder (read-only) given by --run for section G.

Revision after the critique (A1-syndromes-critique.md):
  * G3 uses all 30 Clifford sequences by default (--n-cliffords 30), reports
    the per-sequence slope spread and the first-10 subset, and evaluates the
    leading-order posterior-threshold family with the circuit prior.
  * G4 validates a group-based two-fault routine (order2_fast) against the
    reference loop; G4b applies it to all sequences at m = 2, 4, 6; G4c
    compares the predicted unif-1 fidelity with the baseline.
  * Code capacity: tied pairs with unequal or zero partner rates (C10).
  * P2: kappa decomposition of the final-round hybrid failure (C12).
  * G0 semantics check of the token sequences (C20); G5 catalog
    readout-flip probabilities (C23); G6 logical action of the token
    circuits (C21); G7 final-round schedule variant (C13).
  Sampled-shot checks of the decoders are in A1_sampled_validation.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from fractions import Fraction

import numpy as np

D = 3            # qutrits
N = 5            # data qutrits
NQ = 9           # data + ancilla wires in the circuits

REPORT: dict = {}   # machine-readable results, dumped to JSON at the end


def log(*args):
    print(*args, flush=True)


# =============================================================================
# A. Pauli algebra and the code
# =============================================================================

def pvec(x=None, z=None, n=N):
    """Pauli vector from dicts {qutrit: exponent}."""
    v = np.zeros(2 * n, dtype=np.int64)
    for q, e in (x or {}).items():
        v[q] = e % D
    for q, e in (z or {}).items():
        v[n + q] = e % D
    return v


def symp(a, b, n=N):
    """<A,B> = z_A.x_B - x_A.z_B mod 3 (A B = w^<A,B> B A)."""
    return int((a[n:] @ b[:n] - a[:n] @ b[n:]) % D)


def functional(s, n=N):
    """Row vector f with f.E = <S,E> mod 3."""
    return np.concatenate([s[n:], (-s[:n]) % D]) % D


def weight(v, n=N):
    return int(np.count_nonzero((v[:n] % D != 0) | (v[n:] % D != 0)))


def exp_str(e):
    return "" if e % D == 1 else "^-1"


def plabel(v, n=N):
    """Human-readable label, e.g. 'X0 Z2^-1' or 'X1Z1^-1'."""
    parts = []
    for q in range(n):
        x, z = int(v[q] % D), int(v[n + q] % D)
        if x == 0 and z == 0:
            continue
        s = ""
        if x:
            s += f"X{q}{exp_str(x)}"
        if z:
            s += f"Z{q}{exp_str(z)}"
        parts.append(s)
    return " ".join(parts) if parts else "I"


# Stabilizers as read from src/lrb/code_definitions.py (brief, [code]).
S_Z1 = pvec(z={0: 1, 1: -1, 3: -1})
S_Z2 = pvec(z={1: 1, 2: 1, 4: -1})
S_X1 = pvec(x={0: 1, 1: 1, 2: -1})
S_X2 = pvec(x={1: -1, 3: 1, 4: -1})
STABS = [S_Z1, S_Z2, S_X1, S_X2]
STAB_NAMES = ["S_Z1", "S_Z2", "S_X1", "S_X2"]
X_L = pvec(x={0: 1, 3: 1})
Z_L = pvec(z={0: 1, 2: 1})

# 6 x 10 linear map E -> (sigma_Z1, sigma_Z2, sigma_X1, sigma_X2, a, b)
FUNC = np.array([functional(s) for s in STABS]
                + [functional(Z_L), (-functional(X_L)) % D], dtype=np.int64)


def syn_of(v):
    return tuple(int(t) for t in (FUNC[:4] @ v) % D)


def cos_of(v):
    a, b = (FUNC[4:] @ v) % D
    return int(a), int(b)


def syn_index(s):
    return int(s[0] + 3 * s[1] + 9 * s[2] + 27 * s[3])


def syn_from_index(i):
    return tuple((i // 3 ** j) % 3 for j in range(4))


def coset_label(c):
    a, b = c % 3, c // 3
    parts = []
    if a:
        parts.append("XL" + ("" if a == 1 else "^-1"))
    if b:
        parts.append("ZL" + ("" if b == 1 else "^-1"))
    return " ".join(parts) if parts else "I"


def rank_mod3(M):
    M = np.array(M, dtype=np.int64) % D
    r = 0
    rows, cols = M.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if M[i, c] % D:
                piv = i
                break
        if piv is None:
            continue
        M[[r, piv]] = M[[piv, r]]
        inv = pow(int(M[r, c]), -1, D)
        M[r] = (M[r] * inv) % D
        for i in range(rows):
            if i != r and M[i, c]:
                M[i] = (M[i] - M[i, c] * M[r]) % D
        r += 1
        if r == rows:
            break
    return r


def solve_mod3(A, B):
    """Solve X A = B over Z_3 for X (A: k x m full row rank k, B: l x m)."""
    A = np.array(A, dtype=np.int64) % D
    B = np.array(B, dtype=np.int64) % D
    k, m = A.shape
    # brute force over X rows (k <= 6, 3^k candidates per row)
    cands = np.array(list(itertools.product(range(3), repeat=k)),
                     dtype=np.int64)
    img = (cands @ A) % D
    X = []
    for row in B:
        hit = np.nonzero(np.all(img == row, axis=1))[0]
        if len(hit) != 1:
            raise ValueError("no unique solution")
        X.append(cands[hit[0]])
    return np.array(X, dtype=np.int64)


# ---- frame / conjugation rules (sdim conventions, see sdim/program.py) ------
# Error E before gate U becomes U E U^dagger after it.  The same rules map a
# Pauli operator under conjugation.

def apply_gate(frame, gate, a, b=None, n=N, reduce=True):
    """In-place conjugation of frame (shape (2n,) or (2n, batch))."""
    x, z = frame[:n], frame[n:]
    if gate in ("I", "X", "X_INV", "Z", "Z_INV"):
        return
    if gate == "H":
        t = x[a].copy()
        x[a] = -z[a]
        z[a] = t
    elif gate == "H_INV":
        t = x[a].copy()
        x[a] = z[a]
        z[a] = -t
    elif gate == "P":
        z[a] += x[a]
    elif gate == "P_INV":
        z[a] -= x[a]
    elif gate == "CNOT":
        x[b] += x[a]
        z[a] -= z[b]
    elif gate == "CNOT_INV":
        x[b] -= x[a]
        z[a] += z[b]
    elif gate == "CZ":
        z[b] += x[a]
        z[a] += x[b]
    elif gate == "CZ_INV":
        z[b] -= x[a]
        z[a] -= x[b]
    elif gate == "SWAP":
        t = x[a].copy()
        x[a] = x[b]
        x[b] = t
        t = z[a].copy()
        z[a] = z[b]
        z[b] = t
    else:
        raise ValueError(gate)
    if reduce:
        frame %= D


# Logical gate implementations (brief / code_definitions.py).  Each logical
# token is a list of gate groups (gate, controls, target); "_INV" tokens
# reverse the group order and invert each gate but keep the qudit order
# inside a group, exactly as elementary_single_qudit_logical_operator_gate_
# sequence does.
LOGICAL_GATES = {
    "H": [("SWAP", [2], 3), ("H", [0, 2, 3, 4], None), ("H_INV", [1], None)],
    "P": [("P", [0, 4], None), ("P_INV", [1], None), ("CZ", [2], 3)],
    "X": [("X", [0, 3], None)],
    "Z": [("Z", [0, 2], None)],
    "I": [("I", [0], None)],
}
INV1 = {"H": "H_INV", "H_INV": "H", "P": "P_INV", "P_INV": "P", "X": "X_INV",
        "X_INV": "X", "Z": "Z_INV", "Z_INV": "Z", "CNOT": "CNOT_INV",
        "CNOT_INV": "CNOT", "CZ": "CZ_INV", "CZ_INV": "CZ", "SWAP": "SWAP",
        "I": "I"}


def logical_gate_ops(token):
    """Elementary ops (gate, q[, t]) for one logical token (H, P, X, Z, *_INV)."""
    groups = list(LOGICAL_GATES[token[0]])
    if len(token) > 1:  # _INV: reverse the group list and invert the gates
        groups = [(INV1[g], c, t) for (g, c, t) in reversed(groups)]
    ops = []
    for g, cs, t in groups:
        for c in cs:
            ops.append((g, c) if t is None else (g, c, t))
    return ops


def symplectic_matrix(ops, n=N):
    """Matrix G with (U E U^dag) = G @ E for the op list."""
    cols = []
    for i in range(2 * n):
        f = np.zeros(2 * n, dtype=np.int64)
        f[i] = 1
        for o in ops:
            apply_gate(f, o[0], o[1], o[2] if len(o) > 2 else None, n=n)
        cols.append(f.copy())
    return np.array(cols, dtype=np.int64).T % D


def check_code():
    log("=" * 78)
    log("A. Code checks")
    out = {}
    names = STAB_NAMES + ["X_L", "Z_L"]
    ops = STABS + [X_L, Z_L]
    table = [[symp(p, q) for q in ops] for p in ops]
    log("symplectic products <row,col> (rows/cols: S_Z1 S_Z2 S_X1 S_X2 XL ZL)")
    for nm, row in zip(names, table):
        log(f"  {nm:5s}", row)
    stab_comm = all(table[i][j] == 0 for i in range(4) for j in range(4))
    log_comm = all(table[i][j] == 0 for i in range(4) for j in (4, 5))
    out["stabilizers_commute"] = stab_comm
    out["logicals_commute_with_stabilizers"] = log_comm
    out["ZL_XL"] = table[5][4]
    out["rank_stabilizers"] = rank_mod3(np.array(STABS))
    out["rank_stabilizers_plus_logicals"] = rank_mod3(np.array(ops))
    log("stabilizers commute:", stab_comm, "| logicals commute with S:",
        log_comm, "| <Z_L,X_L> =", table[5][4])
    log("rank(S) =", out["rank_stabilizers"], " rank(S,XL,ZL) =",
        out["rank_stabilizers_plus_logicals"])
    # encoder: ideal |+>_L preparation from code_definitions.py
    enc = [("H_INV", 2), ("H_INV", 3), ("H_INV", 4), ("CNOT_INV", 2, 1),
           ("CNOT", 1, 0), ("CNOT", 3, 0), ("CNOT", 4, 2)]
    G = symplectic_matrix(enc)
    gens = [G @ pvec(z={i: 1}) % D for i in range(N)]
    target = np.array(STABS + [X_L])
    ok = rank_mod3(np.vstack([target, np.array(gens)])) == 5 and \
        rank_mod3(np.array(gens)) == 5
    out["encoder_prepares_plusL"] = bool(ok)
    log("encoder output stabilizers:", [plabel(g) for g in gens])
    log("span(encoder stabilizers) == span(S_Z1,S_Z2,S_X1,S_X2,X_L):", ok)
    # syndromes of the stabilizer generators' check functionals (sanity)
    log("syndrome map rows (functionals on (x0..x4,z0..z4)):")
    for nm, f in zip(STAB_NAMES + ["a(E)", "b(E)"], FUNC):
        log(f"  {nm:5s}", f.tolist())
    REPORT["A_code_checks"] = out
    return out


# =============================================================================
# Enumeration of all 9^5 data errors
# =============================================================================

KS = np.array(list(itertools.product(range(9), repeat=N)), dtype=np.int64)
VECS = np.concatenate([KS % 3, KS // 3], axis=1)        # (59049, 10)
SYN = (VECS @ FUNC[:4].T) % D
COS = (VECS @ FUNC[4:].T) % D
SIDX = SYN[:, 0] + 3 * SYN[:, 1] + 9 * SYN[:, 2] + 27 * SYN[:, 3]
CIDX = COS[:, 0] + 3 * COS[:, 1]
WT = np.count_nonzero(KS, axis=1)


def coset_polys(wq, idx_s=SIDX, idx_c=CIDX, ns=81, nc=9, ks=KS):
    """Integer polynomials P[s, c, k] = coefficient of u^k of
    Pr(syndrome s, coset c) for independent qutrit noise with
    Pr(E_i = k) = wq[i][k] * u (k >= 1), Pr(E_i = 0) = 1 - u * sum_k wq[i][k].
    """
    M = len(ks)
    coef = np.zeros((M, N + 1), dtype=np.int64)
    coef[:, 0] = 1
    for i in range(N):
        wi = np.asarray(wq[i], dtype=np.int64)
        Wi = int(wi[1:].sum())
        k = ks[:, i]
        a = np.where(k == 0, 1, 0)
        b = np.where(k == 0, -Wi, wi[k])
        new = a[:, None] * coef
        new[:, 1:] += b[:, None] * coef[:, :-1]
        coef = new
    P = np.zeros((ns, nc, N + 1), dtype=np.int64)
    np.add.at(P, (idx_s, idx_c), coef)
    tot = P.sum(axis=(0, 1))
    assert tot[0] == 1 and np.all(tot[1:] == 0), tot
    return P


def poly_eval(p, u):
    return float(sum(float(c) * u ** k for k, c in enumerate(p)))


def lex_sign(p, q):
    d = np.asarray(p, dtype=object) - np.asarray(q, dtype=object)
    for t in d:
        if t != 0:
            return 1 if t > 0 else -1
    return 0


def lowest_order(p):
    nz = np.nonzero(np.asarray(p))[0]
    return int(nz[0]) if len(nz) else None


def series_ratio(num, den, order=5):
    """Power series of num/den (integer coefficient lists, den[0] != 0)."""
    num = [Fraction(int(c)) for c in num] + [Fraction(0)] * (order + 1)
    den = [Fraction(int(c)) for c in den] + [Fraction(0)] * (order + 1)
    out = []
    for k in range(order + 1):
        s = num[k] - sum(out[j] * den[k - j] for j in range(k))
        out.append(s / den[0])
    return out


def to_q(coefs_u, Dn):
    return [Fraction(c) / Fraction(Dn) ** k for k, c in enumerate(coefs_u)]


def fmt_series(cq, var="q", nterms=3):
    terms = []
    for k, c in enumerate(cq):
        if c == 0:
            continue
        cs = str(c) if c.denominator != 1 else str(c.numerator)
        if k == 0:
            terms.append(cs)
        else:
            terms.append(f"{cs} {var}" + (f"^{k}" if k > 1 else ""))
        if len(terms) >= nterms:
            break
    return " + ".join(terms).replace("+ -", "- ") if terms else "0"


def analyse_groups(P):
    """Per group (syndrome) leading-order structure of class polynomials."""
    G, H, _ = P.shape
    info = []
    for g in range(G):
        polys = P[g]
        orders = [lowest_order(polys[h]) for h in range(H)]
        present = [o for o in orders if o is not None]
        if not present:
            info.append(None)
            continue
        m = min(present)
        lead = {h: int(polys[h][m]) for h in range(H) if orders[h] == m}
        best = 0
        for h in range(1, H):
            if lex_sign(polys[h], polys[best]) > 0:
                best = h
        ties = [h for h in range(H) if lex_sign(polys[h], polys[best]) == 0]
        maxlead = max(lead.values())
        info.append(dict(
            order=m, lead=lead, mld=best, exact_ties=ties,
            strict_unique=(len(lead) == 1),
            count_unique=(sum(1 for v in lead.values() if v == maxlead) == 1),
            total=polys.sum(axis=0)))
    return info


def strategy_polys(P, info, strategy):
    """Return (P_acc, P_fail_and_acc) integer polynomials for a strategy."""
    G, H, K = P.shape
    acc = np.zeros(K, dtype=object)
    fail = np.zeros(K, dtype=object)
    for g in range(G):
        if info[g] is None:
            continue
        if strategy == "detect":
            ok, dec = (g == 0), 0
        elif strategy == "mld":
            ok, dec = True, info[g]["mld"]
        elif strategy == "hyb_strict":
            ok, dec = info[g]["strict_unique"], info[g]["mld"]
        elif strategy == "hyb_count":
            ok, dec = info[g]["count_unique"], info[g]["mld"]
        else:
            raise ValueError(strategy)
        if not ok:
            continue
        tot = P[g].sum(axis=0).astype(object)
        acc += tot
        fail += tot - P[g][dec].astype(object)
    return acc, fail


def strategy_report(P, Dn, name, qs=(1e-3, 1e-2, 5e-2, 1e-1), order=4,
                    strategies=("detect", "mld", "hyb_strict", "hyb_count")):
    info = analyse_groups(P)
    res = {}
    for st in strategies:
        acc, fail = strategy_polys(P, info, st)
        rej = -acc.copy()
        rej[0] += 1
        rej_q = to_q(list(rej), Dn)
        fail_q = to_q(list(fail), Dn)
        cond_q = to_q(series_ratio(list(fail), list(acc), order), Dn)
        num = {}
        for q in qs:
            u = q / Dn
            pa = poly_eval(acc, u)
            pf = poly_eval(fail, u)
            num[str(q)] = dict(P_acc=pa, P_fail_given_acc=pf / pa,
                               P_fail_and_acc=pf)
        res[st] = dict(
            P_rej_poly_q=[str(c) for c in rej_q],
            P_fail_and_acc_poly_q=[str(c) for c in fail_q],
            P_fail_given_acc_series_q=[str(c) for c in cond_q],
            P_rej_leading=fmt_series(rej_q),
            P_fail_and_acc_leading=fmt_series(fail_q),
            P_fail_given_acc_leading=fmt_series(cond_q),
            numeric=num)
    # finite-q check of the leading-order MLD decision
    changes = {}
    for q in qs:
        u = q / Dn
        n_diff = 0
        for g in range(P.shape[0]):
            if info[g] is None:
                continue
            vals = [poly_eval(P[g][h], u) for h in range(P.shape[1])]
            best = int(np.argmax(vals))
            if abs(vals[best] - vals[info[g]["mld"]]) > 1e-15 * max(1, vals[best]):
                n_diff += 1
        changes[str(q)] = n_diff
    res["mld_decision_changes_vs_leading_order"] = changes
    log(f"--- {name}")
    for st in strategies:
        r = res[st]
        log(f"  {st:10s} P_rej = {r['P_rej_leading']:28s} "
            f"P_fail|acc = {r['P_fail_given_acc_leading']}")
    log("  number of syndromes whose finite-q MLD coset differs from the "
        "leading-order choice:", changes)
    return res, info


def threshold_family(P, Dn, q):
    """Posterior-threshold family at fixed q: accept s iff max_c P(c|s)>=tau."""
    u = q / Dn
    rows = []
    for g in range(P.shape[0]):
        vals = np.array([poly_eval(P[g][h], u) for h in range(P.shape[1])])
        tot = vals.sum()
        if tot <= 0:
            continue
        rows.append((vals.max() / tot, tot, tot - vals.max(), g))
    rows.sort(key=lambda r: -r[0])
    # distinct posterior levels
    out = []
    acc = fail = 0.0
    i = 0
    while i < len(rows):
        level = rows[i][0]
        j = i
        while j < len(rows) and abs(rows[j][0] - level) < 1e-12:
            acc += rows[j][1]
            fail += rows[j][2]
            j += 1
        out.append(dict(tau=level, n_syndromes_at_level=j - i,
                        P_acc=acc, P_fail_given_acc=fail / acc))
        i = j
    return out


# =============================================================================
# B. Syndrome classification (task a)
# =============================================================================

def classify_syndromes():
    log("=" * 78)
    log("B. Syndrome classification (weight <= 2 errors, 9 logical cosets)")
    Ncnt = np.zeros((81, 9, N + 1), dtype=np.int64)
    np.add.at(Ncnt, (SIDX, CIDX, WT), 1)
    rows = []
    errs_by_s = defaultdict(list)
    for idx in np.nonzero(WT <= 2)[0]:
        errs_by_s[int(SIDX[idx])].append(int(idx))
    classes = Counter()
    for s in range(81):
        cnt = Ncnt[s]
        wmin = None
        for w in range(3):
            if cnt[:, w].sum() > 0:
                wmin = w
                break
        cos_at = [c for c in range(9) if wmin is not None and cnt[c, wmin] > 0]
        if wmin == 0:
            cls = "T"            # trivial syndrome
        elif wmin is None:
            cls = "none<=2"
        else:
            lead = [int(cnt[c, wmin]) for c in cos_at]
            if len(cos_at) == 1:
                cls = f"U{wmin}"
            elif len(set(lead)) == 1:
                cls = f"A{wmin}"   # tied, equal leading counts
            else:
                cls = f"A{wmin}*"  # tied at min weight, unequal counts
        classes[cls] += 1
        w1 = {c: [plabel(VECS[i]) for i in errs_by_s[s]
                  if WT[i] == 1 and CIDX[i] == c] for c in range(9)}
        rows.append(dict(
            syndrome=syn_from_index(s), index=s, wmin=wmin, cls=cls,
            cosets_at_wmin=[coset_label(c) for c in cos_at],
            counts_w1={coset_label(c): int(cnt[c, 1]) for c in range(9)
                       if cnt[c, 1]},
            counts_w2={coset_label(c): int(cnt[c, 2]) for c in range(9)
                       if cnt[c, 2]},
            w1_errors={coset_label(c): v for c, v in w1.items() if v},
            w2_errors={coset_label(c): [plabel(VECS[i]) for i in
                                        errs_by_s[s] if WT[i] == 2
                                        and CIDX[i] == c]
                       for c in range(9) if cnt[c, 2]},
        ))
    log("syndrome classes:", dict(classes))
    # weight-1 errors
    w1rows = []
    for idx in np.nonzero(WT == 1)[0]:
        s = int(SIDX[idx])
        r = rows[s]
        partners = [e for c, v in r["w1_errors"].items()
                    for e in v if e != plabel(VECS[idx])]
        q = int(np.nonzero(KS[idx])[0][0])
        k = int(KS[idx][q])
        typ = "pureX" if k in (1, 2) else ("pureZ" if k in (3, 6)
                                            else "mixed")
        w1rows.append(dict(error=plabel(VECS[idx]), qutrit=q, type=typ,
                           syndrome=syn_from_index(s), cls=r["cls"],
                           coset=coset_label(int(CIDX[idx])),
                           confusable_with=partners))
    n_corr = sum(1 for r in w1rows if r["cls"] == "U1")
    log(f"weight-1 errors uniquely decodable at leading order: {n_corr}/40")
    by_q = defaultdict(lambda: Counter())
    for r in w1rows:
        by_q[r["qutrit"]][(r["type"], r["cls"])] += 1
    for q in range(N):
        log(f"  qutrit {q}:", dict(by_q[q]))
    amb = [r for r in w1rows if r["cls"] != "U1"]
    pairs = sorted({tuple(sorted([r["error"]] + r["confusable_with"]))
                    for r in amb})
    log("confusable weight-1 groups:", pairs)
    # weight-2-only syndromes
    w2rows = [r for r in rows if r["wmin"] == 2]
    log(f"syndromes with minimum weight 2: {len(w2rows)};"
        f" classes {Counter(r['cls'] for r in w2rows)}")
    REPORT["B_syndromes"] = rows
    REPORT["B_weight1"] = w1rows
    REPORT["B_classes"] = dict(classes)
    REPORT["B_confusable_groups"] = pairs
    return rows, w1rows, Ncnt


# =============================================================================
# C/D. Code-capacity analysis (tasks b, c)
# =============================================================================

def make_wq(kind):
    """Per-qutrit integer weights (length 9, index k = x + 3z) and D."""
    z = [0] * 9
    dep = [0] + [1] * 8
    pX = [0, 1, 1, 0, 0, 0, 0, 0, 0]
    pZ = [0, 0, 0, 1, 0, 0, 1, 0, 0]
    mixed = [0, 0, 0, 0, 1, 1, 0, 1, 1]
    if kind == "depolarizing":
        return [dep] * N, 8
    if kind == "pure_X":
        return [pX] * N, 2
    if kind == "pure_Z":
        return [pZ] * N, 2
    if kind == "pure_X_or_Z":       # X^a or Z^b only, q/4 each
        return [[0, 1, 1, 1, 0, 0, 1, 0, 0]] * N, 4
    if kind == "mixed_only":         # only X^aZ^b with a,b != 0
        return [mixed] * N, 4
    if kind == "centre_only":
        return [z, dep, z, z, z], 8
    if kind == "corners_only":
        return [dep, z, dep, dep, dep], 8
    # Revision (critique C10): tied pairs with unequal or zero partner rates.
    # Index k = x + 3 z, so X^1 -> k = 1 and X^-1 = X^2 -> k = 2.
    if kind == "unequal_pair_X0_X3inv":      # X0 at 2q/3, X3^-1 at q/3
        return [[0, 2, 0, 0, 0, 0, 0, 0, 0], z, z,
                [0, 0, 1, 0, 0, 0, 0, 0, 0], z], 3
    if kind == "single_pure_X0":             # X0 at q, partner X3^-1 absent
        return [[0, 1, 0, 0, 0, 0, 0, 0, 0], z, z, z, z], 1
    m = re.match(r"qutrit_(\d)_only", kind)
    if m:
        j = int(m.group(1))
        return [dep if i == j else z for i in range(N)], 8
    raise ValueError(kind)


def code_capacity(kinds):
    log("=" * 78)
    log("C/D. Exact code-capacity analysis (all 9^5 configurations)")
    out = {}
    for kind in kinds:
        wq, Dn = make_wq(kind)
        P = coset_polys(wq)
        res, info = strategy_report(P, Dn, f"{kind} (full logical class)")
        # observable-only variant: classes = b only (3 classes)
        # c = a + 3b, so reshape(.., 3, 3) has axes (b, a); sum over a.
        Pb = P.reshape(81, 3, 3, N + 1).sum(axis=2)
        res_b, _ = strategy_report(Pb, Dn, f"{kind} (observable b only)")
        # const0-like: only the X-syndrome (digits 2,3) and b are available
        sx = np.array([syn_from_index(s)[2] + 3 * syn_from_index(s)[3]
                       for s in range(81)])
        Px = np.zeros((9, 3, N + 1), dtype=np.int64)
        for s in range(81):
            Px[sx[s]] += Pb[s]
        res_x, _ = strategy_report(Px, Dn,
                                   f"{kind} (X-syndrome + observable only)")
        ent = dict(D=Dn, full=res, obs_only=res_b, xsyn_obs=res_x)
        if kind == "depolarizing":
            ent["threshold_family"] = {
                str(q): threshold_family(P, Dn, q) for q in (1e-3, 1e-2, 5e-2)}
            ent["threshold_family_obs_only"] = {
                str(q): threshold_family(Pb, Dn, q) for q in (1e-3, 1e-2, 5e-2)}
            # leading-order posterior of each syndrome class
            post = []
            for g in range(81):
                ig = info[g]
                tot = sum(ig["lead"].values())
                post.append(dict(syndrome=syn_from_index(g), order=ig["order"],
                                 lead=ig["lead"],
                                 leading_posterior=str(Fraction(
                                     max(ig["lead"].values()), tot))))
            ent["leading_posteriors"] = post
            log("leading-order posterior values over syndromes:",
                Counter(p["leading_posterior"] for p in post))
        out[kind] = ent
    REPORT["C_code_capacity"] = out
    return out


# =============================================================================
# E. Weight-2 logicals (task d)
# =============================================================================

def logical_reps():
    log("=" * 78)
    log("E. Minimum-weight representatives of the nontrivial logical cosets")
    out = {}
    for c in range(1, 9):
        sel = np.nonzero((SIDX == 0) & (CIDX == c))[0]
        wmin = int(WT[sel].min())
        reps = [plabel(VECS[i]) for i in sel if WT[i] == wmin]
        out[coset_label(c)] = dict(min_weight=wmin, n_reps=len(reps),
                                   reps=reps)
        log(f"  {coset_label(c):10s} min weight {wmin}: {len(reps)} reps,"
            f" e.g. {reps[:6]}")
    sel = np.nonzero((SIDX == 0) & (CIDX != 0) & (WT == 2))[0]
    w2 = [plabel(VECS[i]) for i in sel]
    log("all weight-2 nontrivial logical operators:", w2)
    stab = np.nonzero((SIDX == 0) & (CIDX == 0))[0]
    log("stabilizer group weight distribution:",
        dict(Counter(int(w) for w in WT[stab])))
    out["all_weight2_logicals"] = w2
    out["stabilizer_weight_distribution"] = {
        str(k): v for k, v in Counter(int(w) for w in WT[stab]).items()}
    REPORT["E_logicals"] = out
    return out


# =============================================================================
# F. Logical gates (task f)
# =============================================================================

def strategy_polys_mismatch(P_true, info_dec, strategy):
    """Accept/decide with `info_dec` (a decoder's prior) but score on P_true."""
    G, H, K = P_true.shape
    acc = np.zeros(K, dtype=object)
    fail = np.zeros(K, dtype=object)
    for g in range(G):
        tot = P_true[g].sum(axis=0).astype(object)
        if not np.any(tot != 0):
            continue
        ig = info_dec[g]
        if strategy == "detect":
            ok, dec = (g == 0), 0
        elif ig is None:          # decoder never expects this syndrome
            ok, dec = (strategy == "mld"), 0
        elif strategy == "mld":
            ok, dec = True, ig["mld"]
        elif strategy == "hyb_strict":
            ok, dec = ig["strict_unique"], ig["mld"]
        elif strategy == "hyb_count":
            ok, dec = ig["count_unique"], ig["mld"]
        if not ok:
            continue
        acc += tot
        fail += tot - P_true[g][dec].astype(object)
    return acc, fail


def gate_analysis(w1rows):
    log("=" * 78)
    log("F. Logical gates H_L, P_L (and inverses)")
    out = {}
    wq, Dn = make_wq("depolarizing")
    P0 = coset_polys(wq)
    info0 = analyse_groups(P0)
    w1_by_label = {r["error"]: r for r in w1rows}
    for tok in ("H", "H_INV", "P", "P_INV"):
        ops = logical_gate_ops(tok)
        G = symplectic_matrix(ops)
        # symplectic check
        sym_ok = all(symp(G[:, i], G[:, j]) == symp(np.eye(10, dtype=np.int64)[i],
                                                      np.eye(10, dtype=np.int64)[j])
                     for i in range(10) for j in range(10))
        imgs = [(G @ s) % D for s in STABS]
        stab_ok = all(syn_of(v) == (0, 0, 0, 0) and cos_of(v) == (0, 0)
                      for v in imgs)
        # sigma(G E) = M sigma(E); lambda(G E) = Lam lambda(E) + K sigma(E)
        Fs = FUNC[:4]
        M = solve_mod3(Fs, (Fs @ G) % D)
        XK = solve_mod3(np.vstack([FUNC[4:], Fs]), (FUNC[4:] @ G) % D)
        Lam, K = XK[:, :2], XK[:, 2:]
        log(f"--- {tok}: ops {ops}")
        log(f"  symplectic: {sym_ok}; maps stabilizer group to itself: {stab_ok}")
        log("  U S U^dag:", {nm: plabel(v) for nm, v in zip(STAB_NAMES, imgs)})
        log("  U X_L U^dag =", plabel(G @ X_L % D), "; U Z_L U^dag =",
            plabel(G @ Z_L % D))
        log("  syndrome map M (sigma' = M sigma, order Z1 Z2 X1 X2):",
            M.tolist())
        log("  logical map on (a,b): Lam =", Lam.tolist(), " K =", K.tolist())
        # images of the 40 weight-1 errors
        img_rows = []
        cnt = Counter()
        for idx in np.nonzero(WT == 1)[0]:
            e = VECS[idx]
            e2 = (G @ e) % D
            s2 = syn_index(syn_of(e2))
            c2 = cos_of(e2)[0] + 3 * cos_of(e2)[1]
            ig = info0[s2]
            naive_ok = (ig["mld"] == c2)
            cls = REPORT["B_syndromes"][s2]["cls"]
            w_e2 = weight(e2)
            before = w1_by_label[plabel(e)]["cls"]
            key = (before, f"w{w_e2}", cls,
                   "naive-dec-correct" if naive_ok else "naive-dec-wrong")
            cnt[key] += 1
            img_rows.append(dict(before=plabel(e), before_cls=before,
                                 after=plabel(e2), after_weight=w_e2,
                                 after_syndrome_cls=cls,
                                 naive_decoder_correct=bool(naive_ok)))
        log("  weight-1 errors before the gate -> (class before, weight after,"
            " syndrome class after, naive decoder):")
        for k_, v in sorted(cnt.items()):
            log(f"     {k_}: {v}")
        # exact: iid depolarizing before the gate, decoder after the gate
        V2 = (VECS @ G.T) % D
        S2 = (V2 @ FUNC[:4].T) % D
        C2 = (V2 @ FUNC[4:].T) % D
        sidx2 = S2[:, 0] + 3 * S2[:, 1] + 9 * S2[:, 2] + 27 * S2[:, 3]
        cidx2 = C2[:, 0] + 3 * C2[:, 1]
        P2 = coset_polys(wq, idx_s=sidx2, idx_c=cidx2)
        info2 = analyse_groups(P2)
        res = {}
        for st in ("mld", "hyb_strict"):
            for lab, inf in (("true_prior", info2), ("naive_prior", info0)):
                acc, fail = strategy_polys_mismatch(P2, inf, st)
                rej = -acc.copy()
                rej[0] += 1
                res[f"{st}_{lab}"] = dict(
                    P_rej=fmt_series(to_q(list(rej), Dn)),
                    P_fail_given_acc=fmt_series(to_q(
                        series_ratio(list(fail), list(acc), 4), Dn)))
        # is P2 a permutation of P0?
        perm = sorted(map(tuple, P0.reshape(-1, N + 1).tolist())) == \
            sorted(map(tuple, P2.reshape(-1, N + 1).tolist()))
        log("  depolarizing noise before the gate, decoded after it:")
        for k_, v in res.items():
            log(f"     {k_:22s} P_rej = {v['P_rej']:24s} P_fail|acc = "
                f"{v['P_fail_given_acc']}")
        log("  coset-probability table after the gate is a permutation of the"
            " original:", perm)
        out[tok] = dict(ops=[list(o) for o in ops], symplectic=sym_ok,
                        preserves_stabilizer=stab_ok,
                        stab_images={nm: plabel(v) for nm, v in
                                     zip(STAB_NAMES, imgs)},
                        XL_image=plabel(G @ X_L % D),
                        ZL_image=plabel(G @ Z_L % D),
                        M=M.tolist(), Lam=Lam.tolist(), K=K.tolist(),
                        weight1_images=img_rows,
                        weight1_image_summary={" | ".join(k_): v for k_, v in
                                               cnt.items()},
                        decoding=res, coset_table_is_permutation=perm)
    REPORT["F_gates"] = out
    return out


# =============================================================================
# P. Phenomenological model (task e)
# =============================================================================

W1_PAULIS = [(i, k) for i in range(N) for k in range(1, 9)]


def single_qutrit_vec(i, k):
    v = np.zeros(2 * N, dtype=np.int64)
    v[i] = k % 3
    v[N + i] = k // 3
    return v


def pheno_faults(R, final, late, cls_mode, kappa):
    """List of single faults: (location id, signature tuple, class, weight).

    R noisy rounds; data noise layer before each round (layers 1..R) and an
    optional late layer R+1 after the last noisy round.  final: 'none'
    (observable only, like ordinary LRB circuits), 'X' (perfect final
    X-syndrome, like const=0), 'full' (perfect final full syndrome).
    cls_mode: 'full' -> class (a,b) (index a+3b), 'obs' -> class b.
    Weights are coefficients of q: data Pauli q/8, measurement value
    q_m/2 = kappa q/2.
    """
    nfin = {"none": 0, "X": 2, "full": 4}[final]
    fin_digits = {"none": [], "X": [2, 3], "full": [0, 1, 2, 3]}[final]
    L = 4 * R + nfin
    faults = []
    layers = list(range(1, R + 1)) + ([R + 1] if late else [])
    for r in layers:
        for (i, k) in W1_PAULIS:
            e = single_qutrit_vec(i, k)
            s = np.array(syn_of(e))
            sig = np.zeros(L, dtype=np.int64)
            if r <= R:
                sig[4 * (r - 1):4 * r] = s
            else:
                for t, j in enumerate(fin_digits):
                    sig[4 * R + t] = s[j]
            a, b = cos_of(e)
            c = (a + 3 * b) if cls_mode == "full" else b
            faults.append((("d", r, i), tuple(sig % 3), c, Fraction(1, 8),
                           f"r{r}:{plabel(e)}"))
    for r in range(1, R + 1):
        for j in range(4):
            for v in (1, 2):
                sig = np.zeros(L, dtype=np.int64)
                sig[4 * (r - 1) + j] += v
                if r < R:
                    sig[4 * r + j] -= v
                elif j in fin_digits:
                    sig[4 * R + fin_digits.index(j)] -= v
                faults.append((("m", r, j), tuple(sig % 3), 0,
                               Fraction(kappa) / 2,
                               f"r{r}:meas[{STAB_NAMES[j]}]+{v}"))
    return faults, L


def pheno_analyse(R, final, late, cls_mode, kappa, ncls, want_groups=False):
    faults, L = pheno_faults(R, final, late, cls_mode, kappa)
    zero = tuple([0] * L)
    T1 = defaultdict(lambda: defaultdict(Fraction))
    for loc, sig, c, w, _ in faults:
        T1[sig][c] += w
    # pairs of faults at distinct locations
    T2 = defaultdict(lambda: defaultdict(Fraction))
    pair_list = []
    nF = len(faults)
    sig_arr = np.array([f[1] for f in faults], dtype=np.int64)
    for i in range(nF):
        li = faults[i][0]
        for j in range(i + 1, nF):
            if faults[j][0] == li:
                continue
            sig = tuple(((sig_arr[i] + sig_arr[j]) % 3).tolist())
            c = (faults[i][2] + faults[j][2]) % ncls if cls_mode == "obs" \
                else ((faults[i][2] % 3 + faults[j][2] % 3) % 3
                      + 3 * ((faults[i][2] // 3 + faults[j][2] // 3) % 3))
            w = faults[i][3] * faults[j][3]
            T2[sig][c] += w
            pair_list.append((sig, c, w))

    def decide(sig):
        if sig == zero:
            return True, 0
        t1 = T1.get(sig, {})
        t2 = T2.get(sig, {})
        classes = set(t1) | set(t2)
        best = max(classes, key=lambda h: (t1.get(h, 0), t2.get(h, 0)))
        if t1:
            uniq = sum(1 for h in t1 if t1[h] > 0) == 1
        else:
            uniq = sum(1 for h in t2 if t2[h] > 0) == 1
        return uniq, best

    dec_cache = {}

    def D_(sig):
        if sig not in dec_cache:
            dec_cache[sig] = decide(sig)
        return dec_cache[sig]

    res = dict(det_rej1=Fraction(0), det_fail_acc1=Fraction(0),
               det_fail_acc2=Fraction(0), mld_fail1=Fraction(0),
               hyb_rej1=Fraction(0), hyb_fail_acc1=Fraction(0),
               hyb_fail_acc2=Fraction(0))
    for sig, t in T1.items():
        tot = sum(t.values())
        uniq, best = D_(sig)
        if sig != zero:
            res["det_rej1"] += tot
        else:
            res["det_fail_acc1"] += sum(v for h, v in t.items() if h != 0)
        res["mld_fail1"] += tot - t.get(best, 0)
        if uniq:
            res["hyb_fail_acc1"] += tot - t.get(best, 0)
        else:
            res["hyb_rej1"] += tot
    for sig, c, w in pair_list:
        if sig == zero:
            if c != 0:
                res["det_fail_acc2"] += w
        uniq, best = D_(sig)
        if uniq and c != best:
            res["hyb_fail_acc2"] += w
    groups = []
    if want_groups:
        lab = defaultdict(list)
        for loc, sig, c, w, name in faults:
            lab[sig].append((name, c, w))
        for sig, lst in lab.items():
            cl = {c for _, c, _ in lst}
            if len(cl) > 1 or any(n.find("meas") >= 0 for n, _, _ in lst) \
                    and len(lst) > 1:
                groups.append(dict(signature=sig, explanations=[
                    (n, c, str(w)) for n, c, w in lst]))
    return res, groups


def phenomenological():
    log("=" * 78)
    log("P. Phenomenological model (data q/8 per Pauli per qutrit per round,"
        " measurement q_m/2 per value per digit)")
    out = {}
    configs = [
        ("bulk_full_perfect_final", "full", True, "full"),
        ("ordinary_obs_no_final", "none", False, "obs"),
        ("ordinary_full_no_final", "none", False, "full"),
        ("ordinary_obs_no_final_late", "none", True, "obs"),
        ("xsyndrome_final_obs", "X", True, "obs"),
    ]
    for name, final, late, cmode in configs:
        ncls = 9 if cmode == "full" else 3
        for kappa in (Fraction(1, 4), Fraction(1), Fraction(4)):
            rows = {}
            for R in (1, 2, 3, 4):
                res, groups = pheno_analyse(R, final, late, cmode, kappa,
                                            ncls, want_groups=(R == 2))
                rows[R] = {k: str(v) for k, v in res.items()}
                if R == 2:
                    rows["groups_R2"] = [
                        dict(signature=g["signature"],
                             explanations=g["explanations"])
                        for g in groups]
            out[f"{name}|kappa={kappa}"] = rows
            r1, r4 = rows[1], rows[4]
            log(f"--- {name}, q_m = {kappa} q")
            for R in (1, 2, 3, 4):
                rr = rows[R]
                log(f"   R={R}: det rej {rr['det_rej1']} q, det fail&acc "
                    f"{rr['det_fail_acc1']} q + {rr['det_fail_acc2']} q^2 | MLD fail"
                    f" {rr['mld_fail1']} q | hybrid rej {rr['hyb_rej1']} q, "
                    f"fail&acc {rr['hyb_fail_acc1']} q + {rr['hyb_fail_acc2']} q^2")
    REPORT["P_phenomenological"] = out
    return out


# =============================================================================
# R. Phenomenological model, exact (finite q) for R = 1, 2 by a forward
#    recursion over the 729 hidden states (sigma, a, b).
# =============================================================================

def pheno_exact(q, qm, R, final="full", cls_mode="full"):
    """Exact detection / MLD numbers for the phenomenological model (R <= 2).

    Hidden state h = (sigma(E), coset) with index s + 81*c (729 states).
    Each data layer adds an independent code-capacity error (iid
    depolarizing q per qutrit); round r reports o_r = sigma(E_r) + m_r with
    each digit of m_r nonzero with probability qm (uniform over {1,2}).
    final='full': one more data layer, then a perfect syndrome readout and
    the class of the final error; final='none': class read directly after
    round R (no further syndrome).  cls_mode 'full' (9 cosets) or 'obs'
    (b only).  Forward recursion over all 81^R observation histories.
    """
    wq, Dn = make_wq("depolarizing")
    P = coset_polys(wq)
    u = q / Dn
    pd = np.array([[poly_eval(P[s, c], u) for c in range(9)]
                   for s in range(81)]).T.reshape(-1)   # index s + 81*c
    sv = np.array([syn_from_index(s) for s in range(81)])
    cv = np.array([(c % 3, c // 3) for c in range(9)])
    hs = np.arange(729) % 81
    hc = np.arange(729) // 81
    sum_s = np.array([[syn_index(tuple((sv[a] + sv[b]) % 3)) for b in range(81)]
                      for a in range(81)])
    sum_c = np.array([[((cv[a][0] + cv[b][0]) % 3) + 3 * ((cv[a][1] + cv[b][1]) % 3)
                       for b in range(9)] for a in range(9)])
    add = sum_s[hs[:, None], hs[None, :]] + 81 * sum_c[hc[:, None], hc[None, :]]
    T = np.zeros((729, 729))
    for d in range(729):
        T[np.arange(729), add[:, d]] += pd[d]
    pm = np.array([np.prod([(1 - qm) if t == 0 else qm / 2
                            for t in syn_from_index(m)]) for m in range(81)])
    diff_s = np.array([[syn_index(tuple((sv[o] - sv[s]) % 3)) for s in range(81)]
                       for o in range(81)])
    Like = pm[diff_s[:, hs]]                          # (81 obs, 729 states)
    alpha = np.zeros((1, 729))
    alpha[0, 0] = 1.0
    for r in range(R):
        pred = alpha @ T                              # (H, 729)
        alpha = (pred[:, None, :] * Like[None, :, :]).reshape(-1, 729)
    det_row = 0                                       # all-zero history
    if final == "full":
        alpha = alpha @ T
        A = alpha.reshape(-1, 9, 81)                  # (hist, coset, s_final)
        A = np.transpose(A, (0, 2, 1)).reshape(-1, 9)  # (hist*s_final, coset)
        det_idx = det_row * 81 + 0
    else:
        A = alpha.reshape(-1, 9, 81).sum(axis=2)      # (hist, coset)
        det_idx = det_row
    if cls_mode == "obs":
        A = A.reshape(-1, 3, 3).sum(axis=2)           # coset c = a + 3b
    tot = A.sum(axis=1)
    best = A.max(axis=1)
    res = dict(total=float(tot.sum()), mld_fail=float((tot - best).sum()),
               det_acc=float(tot[det_idx]),
               det_fail=float(tot[det_idx] - A[det_idx, 0]))
    post = best / np.where(tot > 0, tot, 1)
    order = np.argsort(-post)
    res["levels"] = (post[order], tot[order], (tot - best)[order])
    return res


# =============================================================================
# G. Circuit level: generator, parser, batched Pauli-frame propagation
# =============================================================================
# An op is (name, (qudits...), params) with params a dict of strings, as in
# the repository's .chp files.  Noise ops carry prob_scale = coefficient of p.

ONE_Q = {"I", "X", "X_INV", "Z", "Z_INV", "H", "H_INV", "P", "P_INV"}
TWO_Q = {"CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"}


def n1(q, scale, ch="d"):
    return ("N1", (q,), {"noise_channel": ch, "prob_scale": scale})


def n2(a, b, scale=1.0):
    return ("N2", (a, b), {"prob_scale": scale})


def gate_with_noise(name, qs):
    ops = [(name, tuple(qs), {})]
    if name in ONE_Q:
        ops.append(n1(qs[0], 0.1))
    elif name in TWO_Q:
        ops.append(n2(qs[0], qs[1], 1.0))
    return ops


def logical_token_ops_noisy(token):
    out = []
    for o in logical_gate_ops(token):
        out += gate_with_noise(o[0], o[1:])
    return out


def meas_layer(anc, nq=NQ):
    ops = [n1(w, 2.0) for w in range(nq) if w != anc]
    ops += [n1(anc, 1.0), n1(anc, 5.0, "f"), ("M", (anc,), {})]
    return ops


def check_round_ops(nq=NQ):
    """Z_ALT block then X block with generalized SI1000 noise (brief)."""
    ops = []
    for g, a, b in (("CNOT", 0, 5), ("CNOT_INV", 1, 5), ("CNOT_INV", 3, 5)):
        ops += gate_with_noise(g, (a, b))
    ops += meas_layer(5, nq)
    for g, a, b in (("CNOT", 1, 6), ("CNOT", 2, 6), ("CNOT_INV", 4, 6)):
        ops += gate_with_noise(g, (a, b))
    ops += meas_layer(6, nq)
    ops += gate_with_noise("H", (7,))
    for g, a, b in (("CNOT", 7, 0), ("CNOT", 7, 1), ("CNOT_INV", 7, 2)):
        ops += gate_with_noise(g, (a, b))
    ops += gate_with_noise("H_INV", (7,))
    ops += meas_layer(7, nq)
    ops += gate_with_noise("H", (8,))
    for g, a, b in (("CNOT", 8, 3), ("CNOT_INV", 8, 1), ("CNOT_INV", 8, 4)):
        ops += gate_with_noise(g, (a, b))
    ops += gate_with_noise("H_INV", (8,))
    ops += meas_layer(8, nq)
    return ops


def reset_layer_ops():
    ops = [n1(w, 2.0) for w in range(5)]
    for a in (5, 6, 7, 8):
        ops += [("RESET", (a,), {}), n1(a, 2.0, "f")]
    return ops


ENCODER_OPS = [("H_INV", (2,), {}), ("H_INV", (3,), {}), ("H_INV", (4,), {}),
               ("CNOT_INV", (2, 1), {}), ("CNOT", (1, 0), {}),
               ("CNOT", (3, 0), {}), ("CNOT", (4, 2), {})]
TERMINAL_OPS = [("H_INV", (0,), {}), ("H_INV", (3,), {}), ("M", (0,), {}),
                ("M", (3,), {})]
TERMINAL_CONST0_OPS = [("H_INV", (q,), {}) for q in range(5)] + \
    [("M", (q,), {}) for q in range(5)]


def lrb_circuit_ops(forward, inverse_tokens, final_check=True):
    """forward: list of token lists (one per forward Clifford);
    inverse_tokens: flat token list of the inverse block."""
    ops = list(ENCODER_OPS)
    for toks in forward:
        for t in toks:
            ops += logical_token_ops_noisy(t)
        ops += check_round_ops()
        ops += reset_layer_ops()
    for t in inverse_tokens:
        ops += logical_token_ops_noisy(t)
    if final_check:
        ops += check_round_ops()
    ops += TERMINAL_OPS
    nrounds = len(forward) + (1 if final_check else 0)
    ann = []
    for r in range(nrounds):
        for k in range(4):
            ann.append(("DETECTOR", (), {"expr": f"rec[{4 * r + k}]",
                                         "label": f"lrb_stab_r{r}_w{5 + k}"}))
    ann.append(("LOGICAL_OBSERVABLE", (), {
        "expr": f"rec[{4 * nrounds}]+rec[{4 * nrounds + 1}]",
        "label": "lrb_logical"}))
    return ops + ann


def parse_chp(path):
    """Parse a repository .chp file into ops (same indexing as sdim)."""
    with open(path) as fh:
        lines = fh.readlines()
    start = next(i for i, l in enumerate(lines) if l.strip() == "#")
    body = lines[start + 1:]
    if body and body[0].split()[0].upper() == "D":
        body = body[1:]
    ops = []
    for l in body:
        if not l.strip():
            continue
        parts = re.findall(r'\S+="[^"]*"|\S+', l.strip())
        name = parts[0].upper()
        qs = tuple(int(t) for t in parts[1:] if t.isdigit())
        params = {}
        for t in parts[1:]:
            if "=" in t:
                k, v = t.split("=", 1)
                params[k] = v.strip('"')
        ops.append((name, qs, params))
    return ops


def tokens_from_ops(ops):
    """Recover the logical token structure of an LRB circuit file."""
    templates = {t: [(o[0], tuple(o[1:])) for o in logical_gate_ops(t)]
                 for t in ("H", "H_INV", "P", "P_INV", "X", "X_INV", "Z",
                           "Z_INV", "I")}
    gates = [(i, (o[0], o[1])) for i, o in enumerate(ops)
             if o[0] not in ("N1", "N2", "DETECTOR", "LOGICAL_OBSERVABLE")]
    gates = gates[len(ENCODER_OPS):]
    seq = [g for _, g in gates]
    check_gates = [(o[0], o[1]) for o in check_round_ops() if o[0] != "N1"
                   and o[0] != "N2"]
    reset_gates = [(o[0], o[1]) for o in reset_layer_ops() if o[0] != "N1"]
    segments = []
    cur = []
    i = 0
    while i < len(seq):
        if seq[i:i + len(check_gates)] == check_gates:
            i += len(check_gates)
            if seq[i:i + len(reset_gates)] == reset_gates:
                i += len(reset_gates)
                segments.append(("fwd", cur))
                cur = []
            else:
                segments.append(("inv_then_final", cur))
                cur = []
            continue
        matched = False
        for t, tpl in templates.items():
            if seq[i:i + len(tpl)] == tpl:
                cur.append(t)
                i += len(tpl)
                matched = True
                break
        if not matched:
            break
    rest = seq[i:]
    forward = [s for k, s in segments if k == "fwd"]
    inv = [s for k, s in segments if k == "inv_then_final"]
    return forward, (inv[0] if inv else None), rest


def ops_signature(op):
    name, qs, params = op
    if name in ("N1", "N2"):
        return (name, tuple(qs), params.get("noise_channel", "d")
                if name == "N1" else "", float(params.get("prob_scale")))
    if name in ("DETECTOR", "LOGICAL_OBSERVABLE"):
        return (name, params.get("expr"), params.get("label"))
    return (name, tuple(qs))


def validate_generator(path):
    ops = parse_chp(path)
    forward, inv, rest = tokens_from_ops(ops)
    if inv is None:
        return dict(ok=False, reason="could not tokenize")
    gen = lrb_circuit_ops(forward, inv)
    a = [ops_signature(o) for o in ops]
    b = [ops_signature(o) for o in gen]
    first_diff = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]),
                      None)
    ok = (len(a) == len(b)) and first_diff is None
    return dict(ok=ok, n_ops_file=len(a), n_ops_generated=len(b),
                first_difference=first_diff, forward_tokens=forward,
                inverse_tokens=inv)


def parse_expr(expr):
    terms = []
    for sign, coef, idx in re.findall(r'([+-]?)\s*(\d*)\s*\*?\s*rec\[(-?\d+)\]',
                                      expr):
        c = int(coef) if coef else 1
        if sign == "-":
            c = -c
        terms.append((int(idx), c))
    return terms


def propagate_units(ops, nq=NQ, snapshot_at=None):
    """Batched propagation of all unit noise components.

    Returns dict with:
      locs: list of (op_index, name, qudits, channel, prob_scale, col0, ncomp)
      det: (n_det, ncols) detector shifts, obs: (n_obs, ncols),
      final: (2*nq, ncols) frame at the end, meas: (n_meas, ncols).
    """
    locs = []
    col = 0
    for i, (name, qs, params) in enumerate(ops):
        if name in ("N1", "N2"):
            nc = 2 if name == "N1" else 4
            ch = params.get("noise_channel", "d") if name == "N1" else "d"
            locs.append((i, name, qs, ch, float(params.get("prob_scale", 0)),
                         col, nc))
            col += nc
    ncols = col
    F = np.zeros((2 * nq, ncols), dtype=np.int64)
    meas = []
    det_rows, obs_rows, det_labels, obs_labels = [], [], [], []
    li = 0
    snap = None
    n_since = 0
    for i, (name, qs, params) in enumerate(ops):
        if snapshot_at is not None and i == snapshot_at:
            snap = F.copy() % D
        if name == "N1":
            c0 = locs[li][5]
            q = qs[0]
            F[q, c0] += 1
            F[nq + q, c0 + 1] += 1
            li += 1
        elif name == "N2":
            c0 = locs[li][5]
            a, b = qs
            F[a, c0] += 1
            F[nq + a, c0 + 1] += 1
            F[b, c0 + 2] += 1
            F[nq + b, c0 + 3] += 1
            li += 1
        elif name in ONE_Q or name in TWO_Q:
            apply_gate(F, name, qs[0], qs[1] if len(qs) > 1 else None, n=nq,
                       reduce=False)
            n_since += 1
            if n_since >= 16:
                F %= D
                n_since = 0
        elif name == "M":
            F %= D
            n_since = 0
            meas.append(F[qs[0]].copy() % D)
        elif name == "M_X":
            raise ValueError("M_X not expected")
        elif name == "RESET":
            F[qs[0]] = 0
            F[nq + qs[0]] = 0
        elif name in ("DETECTOR", "LOGICAL_OBSERVABLE"):
            terms = parse_expr(params["expr"])
            row = np.zeros(ncols, dtype=np.int64)
            for idx, c in terms:
                row += c * meas[idx]
            if name == "DETECTOR":
                det_rows.append(row % D)
                det_labels.append(params.get("label", ""))
            else:
                obs_rows.append(row % D)
                obs_labels.append(params.get("label", ""))
        else:
            raise ValueError(f"unknown op {name}")
    nd_, no_ = len(det_rows), len(obs_rows)
    return dict(locs=locs, ncols=ncols,
                det=np.array(det_rows, dtype=np.int64).reshape(nd_, ncols),
                obs=np.array(obs_rows, dtype=np.int64).reshape(no_, ncols),
                final=(F % D) if snap is None else snap,
                meas=np.array(meas, dtype=np.int64).reshape(len(meas), ncols),
                det_labels=det_labels, obs_labels=obs_labels)


def pauli_list(name, ch):
    """Nonzero Pauli component vectors of a location and their relative
    weights (fractions of the location probability)."""
    if name == "N1":
        if ch == "d":
            vs = [(x, z) for x in range(3) for z in range(3) if (x, z) != (0, 0)]
        elif ch == "f":
            vs = [(1, 0), (2, 0)]
        elif ch == "p":
            vs = [(0, 1), (0, 2)]
        else:
            raise ValueError(ch)
    else:
        vs = [v for v in itertools.product(range(3), repeat=4) if any(v)]
    return np.array(vs, dtype=np.int64), Fraction(1, len(vs))


def single_faults(prop, extra=None):
    """Expand unit responses into all single faults.

    Returns arrays: resp (n_faults, n_det + n_obs + n_extra), weight (float,
    coefficient of p), loc index per fault, component vectors.
    `extra` is an optional (n_extra, ncols) response matrix (e.g. a perfect
    final syndrome and logical class computed from the final data frame).
    """
    blocks = [prop["det"], prop["obs"]]
    if extra is not None:
        blocks.append(extra)
    U = np.vstack(blocks)                         # (n_resp, ncols)
    resp, wts, locid, comps = [], [], [], []
    for li, (i, name, qs, ch, scale, c0, nc) in enumerate(prop["locs"]):
        if scale == 0:
            continue
        vs, frac = pauli_list(name, ch)
        R = (vs @ U[:, c0:c0 + nc].T) % D         # (nv, n_resp)
        resp.append(R)
        wts.append(np.full(len(vs), scale * float(frac)))
        locid.append(np.full(len(vs), li))
        comps.append(vs if vs.shape[1] == 4 else
                     np.hstack([vs, np.zeros((len(vs), 2), dtype=np.int64)]))
    return (np.vstack(resp), np.concatenate(wts), np.concatenate(locid),
            np.vstack(comps))


def final_extra(prop, nq=NQ):
    """Perfect final syndrome (4) and logical class (a,b) from the final
    data frame of each unit column."""
    F = prop["final"]
    data = np.vstack([F[0:5], F[nq:nq + 5]])      # (10, ncols)
    return (FUNC @ data) % D                      # (6, ncols)


def leading_order(resp, wts, n_sig, cls_cols, ncls_base=3):
    """Group single faults by signature resp[:, :n_sig]; the class is the
    integer formed from the columns cls_cols.  Returns coefficients (of p)
    for detection / MLD / hybrid, the per-fault group index, the per-group
    ambiguity flag and the (n_groups, n_classes) weight table."""
    sig = resp[:, :n_sig].astype(np.int8)
    cls = np.zeros(len(resp), dtype=np.int64)
    for k, c in enumerate(cls_cols):
        cls += resp[:, c] * (ncls_base ** k)
    ncls = ncls_base ** len(cls_cols)
    if n_sig == 0:
        gidx = np.zeros(len(resp), dtype=np.int64)
        uniq = np.zeros((1, 0), dtype=np.int8)
    else:
        sigc = np.ascontiguousarray(sig)
        vv = sigc.view(np.dtype((np.void, sigc.shape[1]))).ravel()
        _, first, gidx = np.unique(vv, return_index=True, return_inverse=True)
        gidx = gidx.reshape(-1)
        uniq = sig[first]
    ng = len(uniq)
    W = np.zeros((ng, ncls))
    np.add.at(W, (gidx, cls), wts)
    zero_g = np.nonzero(~np.any(uniq != 0, axis=1))[0] if n_sig else \
        np.array([0])
    tot = W.sum(axis=1)
    best = W.max(axis=1)
    nclass = (W > 0).sum(axis=1)
    is_zero = np.zeros(ng, dtype=bool)
    is_zero[zero_g] = True
    amb = (nclass > 1) & ~is_zero
    undet = float(W[is_zero][:, 1:].sum()) if is_zero.any() else 0.0
    out = dict(det_rej=float(tot[~is_zero].sum()), undetected_fail=undet,
               mld_fail=float((tot - best)[~is_zero].sum()) + undet,
               hyb_rej=float(tot[amb].sum()), hyb_fail=undet,
               n_faults=int(len(resp)), n_signatures=int(ng),
               total_fault_weight=float(wts.sum()))
    # Revision (critique C8, missing item 2): leading-order posterior-
    # threshold family with the circuit (DEM) prior.  The leading-order
    # posterior of a signature is (largest class weight)/(total weight).
    # Rule tau: accept the zero signature and every signature whose
    # posterior is >= tau, decode to the majority class, reject the rest.
    # tau = 1 is the strict hybrid, tau = 0 is MLD.
    post = np.where(tot > 0, best / np.where(tot > 0, tot, 1.0), 1.0)
    fam = {}
    for tau in TAUS:
        acc = ~is_zero & (post >= tau - 1e-12)
        fam[str(tau)] = dict(rej=float(tot[~is_zero & ~acc].sum()),
                             fail=float((tot - best)[acc].sum()) + undet)
    bins = {}
    for lo, hi in POST_BINS:
        m_ = amb & (post >= lo - 1e-12) & (post < hi - 1e-12)
        bins[f"{lo}-{hi}"] = dict(weight=float(tot[m_].sum()),
                                  minority=float((tot - best)[m_].sum()))
    out["tau_family"] = fam
    out["posterior_bins_of_hybrid_rejected"] = bins
    return out, gidx, amb, W


TAUS = (1.0, 0.99, 0.95, 0.9, 0.8, 0.7, 0.6, 0.5, 0.0)
POST_BINS = ((0.95, 1.0), (0.8, 0.95), (0.6, 0.8), (0.5, 0.6), (0.0, 0.5))


def classify_location(ops, prop):
    """(role, round) for each noise location.  round = index of the first
    check round that comes after the location (ancilla measurements seen
    before the op, divided by 4)."""
    meas_seen = 0
    round_of_op = {}
    for i, (name, qs, params) in enumerate(ops):
        round_of_op[i] = meas_seen // 4
        if name == "M" and qs[0] >= 5:
            meas_seen += 1
    out = []
    for (i, name, qs, ch, scale, c0, nc) in prop["locs"]:
        prev = ops[i - 1] if i > 0 else ("", (), {})
        if name == "N2":
            role = "check_CNOT" if max(qs) >= 5 else "gate_2q"
        elif ch == "f" and prev[0] == "RESET":
            role = "reset_flip"
        elif ch == "f":
            role = "meas_flip"
        elif scale == 0.1:
            role = "gate_1q" if qs[0] < 5 else "anc_H"
        elif scale == 1.0:
            role = "meas_depol"
        elif scale == 2.0:
            j = i
            while j < len(ops) and ops[j][0] == "N1":
                j += 1
            role = ("idle_reset_" if j < len(ops) and ops[j][0] == "RESET"
                    else "idle_meas_") + ("data" if qs[0] < 5 else "anc")
        else:
            role = "other"
        out.append((role, round_of_op[i]))
    return out


# ---- residual-error classification via the code-capacity enumeration -------
MINW = np.full((81, 9), 99, dtype=np.int64)
np.minimum.at(MINW, (SIDX, CIDX), WT)
W1_OF = {}
for _i in np.nonzero(WT == 1)[0]:
    W1_OF[(int(SIDX[_i]), int(CIDX[_i]))] = int(_i)


def residual_type(v):
    """Classify a data Pauli (10-vector) modulo the stabilizer group."""
    s = syn_index(syn_of(v))
    a, b = cos_of(v)
    c = a + 3 * b
    if s == 0 and c == 0:
        return "stabilizer"
    w = int(MINW[s, c])
    if s == 0:
        return f"logical_w{w}"
    if w == 1:
        idx = W1_OF[(s, c)]
        q = int(np.nonzero(KS[idx])[0][0])
        k = int(KS[idx][q])
        typ = "pureX" if k in (1, 2) else ("pureZ" if k in (3, 6) else "mixed")
        amb = REPORT["B_syndromes"][s]["cls"]
        return f"w1_q{q}_{typ}_{amb}"
    return f"w{w}_syn{REPORT['B_syndromes'][s]['cls']}"


def cycle_table():
    """Single faults of one check round + reset (no logical gates)."""
    log("=" * 78)
    log("G1. One check round + reset: exact single-fault table (per unit p)")
    ops = check_round_ops() + reset_layer_ops()
    prop = propagate_units(ops)
    roles = classify_location(ops, prop)
    F = prop["final"]
    out_rows = defaultdict(float)
    per_digit_flip = np.zeros(4)
    per_digit_measonly = np.zeros(4)
    per_digit_reset = np.zeros(4)
    per_qutrit = defaultdict(float)
    per_w1 = defaultdict(float)
    role_tot = defaultdict(float)
    for li, (i, name, qs, ch, scale, c0, nc) in enumerate(prop["locs"]):
        vs, frac = pauli_list(name, ch)
        flips = (vs @ prop["meas"][:, c0:c0 + nc].T) % D          # (nv, 4)
        resid = (vs @ np.vstack([F[0:5], F[NQ:NQ + 5]])[:, c0:c0 + nc].T) % D
        nxt = (vs @ F[5:9, c0:c0 + nc].T) % D                   # reset flips
        w = scale * float(frac)
        for k in range(len(vs)):
            rt = residual_type(resid[k])
            f_now = bool(np.any(flips[k]))
            f_next = bool(np.any(nxt[k]))
            if rt == "stabilizer":
                cat = "meas_only" if (f_now or f_next) else "no_effect"
            else:
                cat = "data+flip" if f_now else "data_only"
            out_rows[(cat, rt if rt != "stabilizer" else "-")] += w
            role_tot[(roles[li][0], cat)] += w
            per_digit_flip += (flips[k] != 0) * w
            if rt == "stabilizer":
                per_digit_measonly += (flips[k] != 0) * w
            per_digit_reset += (nxt[k] != 0) * w
            if rt.startswith("w1_q"):
                per_qutrit[rt] += w
                s_ = syn_index(syn_of(resid[k]))
                a_, b_ = cos_of(resid[k])
                per_w1[W1_OF[(s_, a_ + 3 * b_)]] += w
    tot_by_cat = defaultdict(float)
    for (cat, rt), w in out_rows.items():
        tot_by_cat[cat] += w
    log("totals by category (coefficient of p):",
        {k: round(v, 4) for k, v in tot_by_cat.items()})
    log("this-round flip rate per check (any cause):",
        dict(zip(STAB_NAMES, np.round(per_digit_flip, 4))))
    log("  of which measurement-only (no data residual):",
        dict(zip(STAB_NAMES, np.round(per_digit_measonly, 4))))
    log("next-round flip from reset noise:",
        dict(zip(STAB_NAMES, np.round(per_digit_reset, 4))))
    agg = defaultdict(float)
    for (cat, rt), w in out_rows.items():
        if rt.startswith("w1_q"):
            q = rt.split("_")[1]
            amb = rt.split("_")[-1]
            agg[(q, amb)] += w
        elif rt != "-":
            agg[(rt, "")] += w
    log("data residual (any category) by equivalent weight-1 error:")
    for k_ in sorted(agg):
        log(f"   {k_}: {agg[k_]:.4f}")
    # code-capacity-style estimate with the exact per-round residual rates:
    # tied pairs (A1 syndromes) -> MLD fails on the less likely member,
    # the hybrid rejects both.
    pairs = defaultdict(list)
    for idx in np.nonzero(WT == 1)[0]:
        s_ = int(SIDX[idx])
        if REPORT["B_syndromes"][s_]["cls"] == "A1":
            pairs[s_].append(int(idx))
    mld_est = sum(min(per_w1[i] for i in v) for v in pairs.values())
    hyb_est = sum(sum(per_w1[i] for i in v) for v in pairs.values())
    pair_rates = {" / ".join(plabel(VECS[i]) for i in v):
                  [round(per_w1[i], 4) for i in v] for v in pairs.values()}
    log("tied single-qutrit pairs, per-round residual rates (coeff of p):")
    for k_, v in pair_rates.items():
        log(f"   {k_:18s} {v}")
    log(f"code-capacity-style estimate per round: MLD tie failure "
        f"{mld_est:.4f} p, hybrid rejection of ties {hyb_est:.4f} p")
    REPORT["G1_cycle_tie_estimate"] = dict(
        pair_rates=pair_rates, mld_tie_failure=mld_est,
        hybrid_tie_rejection=hyb_est,
        per_w1_rates={plabel(VECS[i]): w for i, w in per_w1.items()})
    REPORT["G1_cycle"] = dict(
        by_category={f"{c}|{r}": v for (c, r), v in out_rows.items()},
        totals=dict(tot_by_cat),
        flip_rate_per_check=dict(zip(STAB_NAMES, per_digit_flip.tolist())),
        measonly_flip_per_check=dict(zip(STAB_NAMES,
                                         per_digit_measonly.tolist())),
        reset_flip_per_check=dict(zip(STAB_NAMES, per_digit_reset.tolist())),
        data_residual_w1={f"{q}|{a}": v for (q, a), v in agg.items()},
        role_by_category={f"{r}|{c}": v for (r, c), v in role_tot.items()})
    return REPORT["G1_cycle"]


def convention_check():
    """Propagate every unit data Pauli through a noiseless check round and
    through the two terminal readouts; compare with the algebraic maps."""
    def noiseless(ops):
        return [o for o in ops if o[0] not in ("N1", "N2")]
    out = {}
    blocks = dict(
        check_round=(noiseless(check_round_ops()), None),
        terminal=(TERMINAL_OPS, None),
        terminal_const0=(TERMINAL_CONST0_OPS, None))
    for name, (ops, _) in blocks.items():
        ok = True
        for i in range(2 * N):
            e = np.zeros(2 * NQ, dtype=np.int64)
            q = i % N
            e[q if i < N else NQ + q] = 1
            F = e.copy()
            shifts = []
            for (g, qs, params) in ops:
                if g == "M":
                    shifts.append(int(F[qs[0]] % D))
                else:
                    apply_gate(F, g, qs[0], qs[1] if len(qs) > 1 else None,
                               n=NQ)
            v = np.concatenate([e[:N], e[NQ:NQ + N]])
            if name == "check_round":
                ok &= tuple(shifts) == syn_of(v)
            elif name == "terminal":
                ok &= (shifts[0] + shifts[1]) % D == cos_of(v)[1]
            else:
                x = shifts
                det = ((x[0] + x[1] - x[2]) % D, (-x[1] + x[3] - x[4]) % D)
                s = syn_of(v)
                ok &= det == ((-s[2]) % D, (-s[3]) % D) and \
                    (x[0] + x[3]) % D == cos_of(v)[1]
        out[name] = bool(ok)
    log("convention check (ancilla shifts == <S,E>; terminal x0+x3 == b(E);"
        " const0 detectors == -sigma_X):", out)
    REPORT["G0_convention_check"] = out
    return out


def terminal_start(ops):
    i = next(k for k, o in enumerate(ops) if o[0] in ("DETECTOR",
                                                      "LOGICAL_OBSERVABLE"))
    if ops[i - 1] == ("M", (3,), {}) and ops[i - 2] == ("M", (0,), {}):
        return i - len(TERMINAL_OPS)
    return i - len(TERMINAL_CONST0_OPS)


def analyse_circuit(ops, want_breakdown=True):
    t0 = terminal_start(ops)
    prop = propagate_units(ops, snapshot_at=t0)
    extra = final_extra(prop)
    resp, wts, locid, comps = single_faults(prop, extra=extra)
    nd = prop["det"].shape[0]
    # sanity: b of the final frame equals the observable shift
    assert np.array_equal(resp[:, nd], resp[:, nd + 6]), "b != observable"
    v1, g1, amb1, W1 = leading_order(resp, wts, nd, [nd])
    resp2 = np.hstack([resp[:, :nd], resp[:, nd + 1:nd + 5],
                       resp[:, nd + 5:nd + 7]])
    v2, g2, amb2, W2 = leading_order(resp2, wts, nd + 4, [nd + 4, nd + 5])
    res = dict(n_detectors=nd, V1_as_run=v1, V2_perfect_final_fullclass=v2)
    if want_breakdown:
        roles = classify_location(ops, prop)
        role_f = np.array([roles[l][0] for l in locid])
        rnd_f = np.array([roles[l][1] for l in locid])
        cls1 = resp[:, nd]
        zero_g = ~np.any(resp[:, :nd] != 0, axis=1)
        best1 = W1.argmax(axis=1)
        wrong = (cls1 != best1[g1])
        bd = defaultdict(lambda: defaultdict(float))
        for name_, mask in (("detected", ~zero_g),
                            ("undetected_obs_flip", zero_g & (cls1 != 0)),
                            ("ambiguous_signature", amb1[g1]),
                            ("mld_wrong", wrong)):
            for r_, w_ in zip(role_f[mask], wts[mask]):
                bd[name_][r_] += w_
        res["V1_breakdown_by_role"] = {k: dict(v) for k, v in bd.items()}
        nr = int(rnd_f.max()) + 1
        byr = {}
        for name_, mask in (("detected", ~zero_g),
                            ("ambiguous_signature", amb1[g1]),
                            ("mld_wrong", wrong),
                            ("undetected_obs_flip", zero_g & (cls1 != 0))):
            arr = np.zeros(nr)
            np.add.at(arr, rnd_f[mask], wts[mask])
            byr[name_] = arr.tolist()
        res["V1_by_round"] = byr
        # same breakdown for V2 (perfect final syndrome, full class)
        cls2 = resp2[:, nd + 4] + 3 * resp2[:, nd + 5]
        zero2 = ~np.any(resp2[:, :nd + 4] != 0, axis=1)
        best2 = W2.argmax(axis=1)
        wrong2 = (cls2 != best2[g2]) & ~zero2
        bd2 = defaultdict(lambda: defaultdict(float))
        byr2 = {}
        for name_, mask in (("detected", ~zero2),
                            ("ambiguous_signature", amb2[g2]),
                            ("mld_wrong", wrong2)):
            for r_, w_ in zip(role_f[mask], wts[mask]):
                bd2[name_][r_] += w_
            arr = np.zeros(nr)
            np.add.at(arr, rnd_f[mask], wts[mask])
            byr2[name_] = arr.tolist()
        res["V2_breakdown_by_role"] = {k: dict(v) for k, v in bd2.items()}
        res["V2_by_round"] = byr2
    return res


def catalog_token_stats(worktree):
    """Average logical-token counts per catalog Clifford (read-only parse)."""
    path = os.path.join(worktree, "src", "lrb", "code_definitions.py")
    txt = open(path).read()
    blk = txt[txt.index("single_qutrit_cliffords_exhaustive_with_S_strings = ["):]
    blk = blk[:blk.index("]")]
    strs = re.findall(r'"([A-Z]+)"', blk)
    cnt = Counter()
    for s in strs:
        s2 = s.replace("S", "P")
        if s2 == "I":
            cnt["I"] += 1
        for ch in s2:
            if ch != "I":
                cnt[ch] += 1
    n = len(strs)
    return dict(n_strings=n, n_distinct=len(set(strs)),
                mean_tokens={k: v / n for k, v in cnt.items()})


def gate_noise_per_token():
    """Single-fault weights (coefficient of p) of one noisy logical token,
    and the residual-type split of its data errors (code-capacity view)."""
    out = {}
    for tok in ("H", "P", "X", "Z", "I"):
        ops = logical_token_ops_noisy(tok)
        prop = propagate_units(ops, nq=5)
        agg = defaultdict(float)
        for (i, name, qs, ch, scale, c0, nc) in prop["locs"]:
            vs, frac = pauli_list(name, ch)
            F = prop["final"][:, c0:c0 + nc]
            resid = (vs @ F.T) % D
            for k in range(len(vs)):
                rt = residual_type(resid[k])
                key = rt if not rt.startswith("w1_q") else \
                    ("w1_" + rt.split("_")[-1])
                agg[key] += scale * float(frac)
        out[tok] = dict(agg)
    return out
def dem_crosscheck(path, worktree_src):
    """Compare own unit responses with the repository DEM (needs sdim)."""
    sys.path.insert(0, worktree_src)
    from lrb.correction.compact_dem import read_run_circuit as read_circuit
    from lrb.correction.compact_dem import build_compact_detector_error_model
    circ = read_circuit(path)
    model = build_compact_detector_error_model(circ)
    ops = parse_chp(path)
    prop = propagate_units(ops)
    by_op = {l[0]: l for l in prop["locs"]}
    n_cmp = n_bad = 0
    for loc in model.locations:
        mine = by_op[loc.operation_index]
        assert mine[1] == loc.gate and mine[2][0] == loc.qudit
        for comp, terms in enumerate(loc.component_terms):
            col = mine[5] + comp
            d = np.zeros(prop["det"].shape[0], dtype=np.int64)
            for idx, v in terms.detector:
                d[idx] = v
            o = np.zeros(prop["obs"].shape[0], dtype=np.int64)
            for idx, v in terms.logical:
                o[idx] = v
            n_cmp += 1
            if not (np.array_equal(d, prop["det"][:, col])
                    and np.array_equal(o, prop["obs"][:, col])):
                n_bad += 1
    return dict(circuit=path, components_compared=n_cmp, mismatches=n_bad,
                dem_locations=len(model.locations),
                own_locations=sum(1 for l in prop["locs"] if l[4] > 0))


def memory_windows(Rmax=5):
    """Circuits with R noisy check rounds (each followed by reset), no
    logical gates, then the final noisy check round and the ordinary
    terminal readout.  Leading-order coefficients for the as-run view (V1)
    and the perfect-final-syndrome/full-class view (V2)."""
    log("=" * 78)
    log("G2. Memory windows (identity Cliffords, SI1000 check noise only)")
    rows = []
    for R in range(Rmax + 1):
        ops = lrb_circuit_ops([[] for _ in range(R)], [])
        r = analyse_circuit(ops, want_breakdown=(R in (0, 3)))
        rows.append(dict(R=R, V1=r["V1_as_run"],
                         V2=r["V2_perfect_final_fullclass"],
                         breakdown=r.get("V1_breakdown_by_role"),
                         breakdown_V2=r.get("V2_breakdown_by_role"),
                         by_round_V1=r.get("V1_by_round"),
                         by_round_V2=r.get("V2_by_round")))
        v1, v2 = r["V1_as_run"], r["V2_perfect_final_fullclass"]
        log(f"  R={R}: V1 det_rej {v1['det_rej']:.3f} undet {v1['undetected_fail']:.3f}"
            f" mld {v1['mld_fail']:.3f} hyb_rej {v1['hyb_rej']:.3f} | V2 det_rej "
            f"{v2['det_rej']:.3f} mld {v2['mld_fail']:.3f} hyb_rej {v2['hyb_rej']:.3f}")
    inc = {}
    for key in ("det_rej", "mld_fail", "hyb_rej", "undetected_fail"):
        for V in ("V1", "V2"):
            vals = [r[V][key] for r in rows]
            inc[f"{V}_{key}"] = [b - a for a, b in zip(vals, vals[1:])]
    log("  per-round increments (R -> R+1):")
    for k_, v in inc.items():
        log(f"     {k_:20s} {np.round(v, 4).tolist()}")
    REPORT["G2_memory_windows"] = dict(rows=rows, increments=inc)
    return rows, inc


def baseline_csv(run, p_idx, policy, key):
    path = os.path.join(run, "results", "LRB", str(p_idx), policy, f"{key}.csv")
    rows = {}
    with open(path) as fh:
        for line in fh:
            parts = line.strip().split(",")
            rows[parts[0]] = [float(t) for t in parts[1:]]
    return rows


def lrb_circuits(run, cliffs, depths_idx, p_idx_files=4):
    log("=" * 78)
    log("G3. Baseline LRB circuits (own propagation, leading order in p)")
    depths = [int(t) for t in open(os.path.join(run, "depths.txt")).read()
              .strip().split(",")]
    res = defaultdict(list)
    res0 = defaultdict(list)
    t0 = time.time()
    for c in cliffs:
        for di in depths_idx:
            path = os.path.join(run, "experiments", "LRB", str(c),
                                str(p_idx_files), f"{di}.chp")
            ops = parse_chp(path)
            r = analyse_circuit(ops, want_breakdown=(c == cliffs[0]))
            res[di].append(r)
            path0 = os.path.join(run, "experiments", "LRB_const0", str(c),
                                 str(p_idx_files), f"{di}.chp")
            ops0 = parse_chp(path0)
            prop0 = propagate_units(ops0)
            if prop0["ncols"] == 0:       # depth 0: no noisy operation
                v0 = dict(det_rej=0.0, undetected_fail=0.0, mld_fail=0.0,
                          hyb_rej=0.0, hyb_fail=0.0)
            else:
                resp0, w0, _, _ = single_faults(prop0)
                nd0 = prop0["det"].shape[0]
                v0, _, _, _ = leading_order(resp0, w0, nd0, [nd0])
            res0[di].append(v0)
    log(f"  analysed {len(cliffs) * len(depths_idx)} LRB + const0 circuits in"
        f" {time.time() - t0:.1f} s ({len(cliffs)} Clifford sequences)")
    # baseline data at p = 1e-4 (p index 1) for comparison
    b_unif = baseline_csv(run, 1, "unif_check_data", 1)
    b_c0 = baseline_csv(run, 1, "const_check_data", 0)
    p = 1e-4
    table = _lrb_table(res, res0, depths, depths_idx, b_unif, b_c0, p,
                       list(range(len(cliffs))))
    fits = _lrb_fits(table)
    for row in table:
        log(f"  m={row['depth']:2d}: det_rej {row['det_rej']:8.2f} (baseline -ln acc/p "
            f"{row['baseline_unif1_minuslog_acc_over_p']:8.2f}) undet "
            f"{row['undetected']:.3f} (baseline (1-F)/1.5p "
            f"{row['baseline_unif1_infid_over_1p5p']:.3f}) | MLD {row['mld_fail']:7.2f}"
            f" hyb_rej {row['hyb_rej']:7.2f} | const0 det_rej "
            f"{row['const0_det_rej']:7.2f} (baseline {row['baseline_const0_minuslog_acc_over_p']:7.2f})"
            f" mld {row['const0_mld_fail']:6.2f} hyb_rej {row['const0_hyb_rej']:6.2f}")
    log(f"  linear fits c(m) = c0 + c1 m (m >= 2), {len(cliffs)} sequences:")
    for k_, v in fits.items():
        log(f"     {k_:38s} c0 = {v['intercept']:9.3f}  c1 = {v['slope_per_depth']:8.3f}")
    # Revision (critique C17): per-sequence slope spread, and the first-10
    # subset used in the first version of the notes.
    ms = np.array([depths[di] for di in depths_idx])
    sel = ms >= 2
    A_ = np.vstack([np.ones(sel.sum()), ms[sel]]).T
    spread = {}
    for key, V in (("det_rej", "V1_as_run"), ("mld_fail", "V1_as_run"),
                   ("hyb_rej", "V1_as_run"), ("mld_fail", "V2_perfect_final_fullclass"),
                   ("hyb_rej", "V2_perfect_final_fullclass")):
        sl = []
        for ci in range(len(cliffs)):
            ys = np.array([res[di][ci][V][key] for di in depths_idx])
            sl.append(float(np.linalg.lstsq(A_, ys[sel], rcond=None)[0][1]))
        nm = ("V2_" if V.startswith("V2") else "") + key
        spread[nm] = _spread(sl)
    for key in ("det_rej", "mld_fail", "hyb_rej"):
        sl = []
        for ci in range(len(cliffs)):
            ys = np.array([res0[di][ci][key] for di in depths_idx])
            sl.append(float(np.linalg.lstsq(A_, ys[sel], rcond=None)[0][1]))
        spread["const0_" + key] = _spread(sl)
    log("  per-sequence slopes (mean, std over sequences, SE of the mean, min, max):")
    for k_, v in spread.items():
        log(f"     {k_:24s} {v['mean']:9.3f} {v['std']:8.3f} {v['se']:7.3f}"
            f" {v['min']:9.3f} {v['max']:9.3f}")
    subset = None
    if len(cliffs) > 10:
        t10 = _lrb_table(res, res0, depths, depths_idx, b_unif, b_c0, p,
                         list(range(10)))
        subset = _lrb_fits(t10)
        log("  first-10-sequence subset slopes (first version of the notes):",
            {k: round(v["slope_per_depth"], 3) for k, v in subset.items()})
    ratio = fits["mld_fail"]["slope_per_depth"] / fits["V2_mld_fail"]["slope_per_depth"]
    log(f"  ratio V1/V2 MLD slope: {ratio:.4f}")
    # Revision (critique C8, missing item 2): threshold family per depth.
    tau_tab = {}
    for V in ("V1_as_run", "V2_perfect_final_fullclass"):
        rows_ = []
        for di in depths_idx:
            ent = dict(depth=depths[di])
            for tau in TAUS:
                ent[f"rej_{tau}"] = float(np.mean([r[V]["tau_family"][str(tau)]["rej"]
                                                   for r in res[di]]))
                ent[f"fail_{tau}"] = float(np.mean([r[V]["tau_family"][str(tau)]["fail"]
                                                    for r in res[di]]))
            for lo, hi in POST_BINS:
                k_ = f"{lo}-{hi}"
                ent[f"bin_{k_}"] = float(np.mean(
                    [r[V]["posterior_bins_of_hybrid_rejected"][k_]["weight"]
                     for r in res[di]]))
            rows_.append(ent)
        tf = {}
        for tau in TAUS:
            for q_ in ("rej", "fail"):
                ys = np.array([e[f"{q_}_{tau}"] for e in rows_])
                c_ = np.linalg.lstsq(A_, ys[sel], rcond=None)[0]
                tf[f"{q_}_{tau}"] = dict(intercept=float(c_[0]), slope_per_depth=float(c_[1]))
        tau_tab[V] = dict(per_depth=rows_, fits=tf)
        log(f"  threshold family ({V}); per-depth slopes of rejection / first-order failure:")
        for tau in TAUS:
            log(f"     tau={tau:4.2f}: rej {tf[f'rej_{tau}']['slope_per_depth']:8.3f} p"
                f" (c0 {tf[f'rej_{tau}']['intercept']:7.2f}) | fail "
                f"{tf[f'fail_{tau}']['slope_per_depth']:7.3f} p (c0 "
                f"{tf[f'fail_{tau}']['intercept']:6.2f})")
        last = rows_[-1]
        log(f"     m={last['depth']}: strict-hybrid rejection {last['rej_1.0']:.2f} p split by"
            " leading-order posterior:",
            {k[4:]: round(v, 2) for k, v in last.items() if k.startswith("bin_")})
    REPORT["G3_lrb_circuits"] = dict(
        cliffords=list(cliffs), table=table, fits=fits,
        per_sequence_slope_spread=spread, subset10_fits=subset,
        V1_over_V2_mld_slope=ratio, threshold_family=tau_tab,
        per_sequence={str(depths[di]): [dict(
            V1={k: r["V1_as_run"][k] for k in ("det_rej", "mld_fail", "hyb_rej",
                                               "undetected_fail")},
            V2={k: r["V2_perfect_final_fullclass"][k] for k in ("mld_fail", "hyb_rej")},
            const0={k: r0[k] for k in ("det_rej", "mld_fail", "hyb_rej")})
            for r, r0 in zip(res[di], res0[di])] for di in depths_idx},
        depth0_breakdown=res[depths_idx[0]][0].get("V1_breakdown_by_role"),
        breakdown_by_depth={str(depths[di]): res[di][0].get("V1_breakdown_by_role")
                            for di in depths_idx},
        by_round_first_clifford={str(depths[di]): res[di][0].get("V1_by_round")
                                 for di in depths_idx})
    return table, fits


def _spread(sl):
    sl = np.array(sl)
    return dict(mean=float(sl.mean()), std=float(sl.std(ddof=1)),
                se=float(sl.std(ddof=1) / np.sqrt(len(sl))), min=float(sl.min()),
                max=float(sl.max()), n=int(len(sl)))


def _lrb_fits(table):
    fits = {}
    ms = np.array([r["depth"] for r in table])
    sel = ms >= 2
    for key in ("det_rej", "mld_fail", "hyb_rej", "undetected", "V2_mld_fail",
                "V2_hyb_rej", "const0_det_rej", "const0_mld_fail",
                "const0_hyb_rej", "baseline_unif1_minuslog_acc_over_p",
                "baseline_const0_minuslog_acc_over_p"):
        ys = np.array([r[key] for r in table])
        A_ = np.vstack([np.ones(sel.sum()), ms[sel]]).T
        coef, *_ = np.linalg.lstsq(A_, ys[sel], rcond=None)
        fits[key] = dict(intercept=float(coef[0]), slope_per_depth=float(coef[1]))
    return fits


def _lrb_table(res, res0, depths, depths_idx, b_unif, b_c0, p, cidx):
    table = []
    for di in depths_idx:
        m = depths[di]
        avg = lambda key, V="V1_as_run": float(np.mean([res[di][c][V][key]
                                                        for c in cidx]))
        a0 = lambda key: float(np.mean([res0[di][c][key] for c in cidx]))
        rej_b = b_unif["Rejected Runs"][di]
        fid_b = b_unif["Fidelity averages"][di]
        row = dict(
            depth=m,
            det_rej=avg("det_rej"), undetected=avg("undetected_fail"),
            mld_fail=avg("mld_fail"), hyb_rej=avg("hyb_rej"),
            V2_mld_fail=avg("mld_fail", "V2_perfect_final_fullclass"),
            V2_hyb_rej=avg("hyb_rej", "V2_perfect_final_fullclass"),
            V2_det_rej=avg("det_rej", "V2_perfect_final_fullclass"),
            const0_det_rej=a0("det_rej"), const0_undetected=a0("undetected_fail"),
            const0_mld_fail=a0("mld_fail"), const0_hyb_rej=a0("hyb_rej"),
            baseline_unif1_rej_p1e4=rej_b,
            baseline_unif1_minuslog_acc_over_p=-np.log(1 - rej_b) / p,
            baseline_unif1_infid_over_1p5p=(1 - fid_b) / 1.5 / p,
            baseline_const0_minuslog_acc_over_p=-np.log(
                1 - b_c0["Rejected Runs"][di]) / p,
            baseline_const0_infid_over_1p5p=(1 - b_c0["Fidelity averages"][di])
            / 1.5 / p)
        table.append(row)
    return table


def order2_analysis(resp, wts, locid, loc_prob, n_sig, cls_cols):
    """Second-order (two-fault) coefficients for detection and the strict
    hybrid decoder, from all pairs of single faults at distinct locations.

    Returns the order-1 and order-2 coefficients of P(fail | accepted) for
    detection (accept iff signature 0) and for the hybrid decoder (accept a
    signature iff all single-fault explanations share one class; decide that
    class; signatures with no single-fault explanation are accepted only if
    all their two-fault explanations share a class, which cannot fail at
    second order).  Class arithmetic is digit-wise mod 3."""
    sig = resp[:, :n_sig].astype(np.int64)
    pow3 = 3 ** np.arange(n_sig, dtype=np.int64)
    assert n_sig <= 39
    key = sig @ pow3
    cl = resp[:, cls_cols].astype(np.int64)
    ncl = 3 ** len(cls_cols)
    cpow = 3 ** np.arange(len(cls_cols), dtype=np.int64)
    cidx = cl @ cpow
    uk, inv = np.unique(key, return_inverse=True)
    W1 = np.zeros((len(uk), ncl))
    np.add.at(W1, (inv, cidx), wts)
    nclass1 = (W1 > 0).sum(axis=1)
    best1 = W1.argmax(axis=1)
    zero_pos = np.searchsorted(uk, 0)
    has_zero = zero_pos < len(uk) and uk[zero_pos] == 0
    acc1 = (nclass1 == 1)
    if has_zero:
        acc1[zero_pos] = True
        best1[zero_pos] = 0
    # order-1 quantities
    undet = wts[(key == 0) & (cidx != 0)].sum()
    det_rej1 = wts[key != 0].sum()
    hyb_rej1 = wts[~acc1[inv]].sum()
    hyb_fail1 = wts[acc1[inv] & (cidx != best1[inv])].sum()
    T = float(sum(loc_prob))
    # pairs
    A2_det = 0.0
    A2_hyb = 0.0
    n = len(resp)
    for i in range(n - 1):
        j = np.arange(i + 1, n)
        j = j[locid[j] != locid[i]]
        if len(j) == 0:
            continue
        k2 = ((sig[i] + sig[j]) % 3) @ pow3
        c2 = ((cl[i] + cl[j]) % 3) @ cpow
        w2 = wts[i] * wts[j]
        z = (k2 == 0)
        A2_det += w2[z & (c2 != 0)].sum()
        pos = np.searchsorted(uk, k2)
        pos = np.minimum(pos, len(uk) - 1)
        inT = uk[pos] == k2
        good = inT & acc1[pos] & ~z
        # zero signature: always accepted with class 0 (also by the hybrid)
        A2_hyb += w2[good & (c2 != best1[pos])].sum() + w2[z & (c2 != 0)].sum()
    # order-2 coefficient of P(fail|acc) including the depletion terms of the
    # order-1 failing single faults
    f1_mask_det = (key == 0) & (cidx != 0)
    f1_mask_hyb = acc1[inv] & (cidx != best1[inv])
    dep_det = float(sum(wts[f] * (T - loc_prob[locid[f]])
                        for f in np.nonzero(f1_mask_det)[0]))
    dep_hyb = float(sum(wts[f] * (T - loc_prob[locid[f]])
                        for f in np.nonzero(f1_mask_hyb)[0]))
    out = dict(
        det_rej1=float(det_rej1), det_fail_given_acc_1=float(undet),
        det_fail_given_acc_2=float(A2_det - dep_det + undet * det_rej1),
        hyb_rej1=float(hyb_rej1), hyb_fail_given_acc_1=float(hyb_fail1),
        hyb_fail_given_acc_2=float(A2_hyb - dep_hyb + hyb_fail1 * hyb_rej1),
        pairs_det=float(A2_det), pairs_hyb=float(A2_hyb), n_faults=int(n))
    return out


def order2_circuit(ops, views=("V1", "V2"), fast=False):
    t0 = terminal_start(ops)
    prop = propagate_units(ops, snapshot_at=t0)
    extra = final_extra(prop)
    resp, wts, locid, comps = single_faults(prop, extra=extra)
    nd = prop["det"].shape[0]
    loc_prob = np.array([l[4] for l in prop["locs"]])
    fn = order2_fast if fast else order2_analysis
    out = {}
    if "V1" in views:
        out["V1"] = fn(resp, wts, locid, loc_prob, nd, [nd])
    if "V2" in views:
        resp2 = np.hstack([resp[:, :nd], resp[:, nd + 1:nd + 5],
                           resp[:, nd + 5:nd + 7]])
        out["V2"] = fn(resp2, wts, locid, loc_prob, nd + 4, [nd + 4, nd + 5])
    return out


def order2_sweep(run):
    log("=" * 78)
    log("G4. Second-order (two-fault) coefficients, detection vs hybrid")
    res = {}
    t0 = time.time()
    for R in (0, 1, 2):
        ops = lrb_circuit_ops([[] for _ in range(R)], [])
        res[f"memory_R{R}"] = order2_circuit(ops)
        log(f"  memory R={R}:", {v: {k: round(x, 3) for k, x in d.items()}
                                 for v, d in res[f"memory_R{R}"].items()})
    if run and os.path.isdir(run):
        for c in (0, 1):
            for di in (0, 1):
                path = os.path.join(run, "experiments", "LRB", str(c), "4",
                                    f"{di}.chp")
                res[f"lrb_c{c}_d{di}"] = order2_circuit(parse_chp(path))
                log(f"  LRB cliff {c} depth idx {di}:",
                    {v: {k: round(x, 3) for k, x in d.items()}
                     for v, d in res[f"lrb_c{c}_d{di}"].items()})
    log(f"  order-2 sweep took {time.time() - t0:.1f} s")
    inc = {}
    for V in ("V1", "V2"):
        for key in ("det_fail_given_acc_2", "hyb_fail_given_acc_2"):
            vals = [res[f"memory_R{R}"][V][key] for R in (0, 1, 2)]
            inc[f"{V}_{key}"] = [b - a for a, b in zip(vals, vals[1:])]
    log("  memory per-round increments of the order-2 coefficients:", inc)
    # Revision: the group-based routine order2_fast must reproduce the
    # reference per-fault loop on every circuit above before it is used.
    maxdiff = 0.0
    for name_ in res:
        if name_.startswith("memory_R"):
            R = int(name_[-1])
            ops = lrb_circuit_ops([[] for _ in range(R)], [])
        else:
            c, di = int(name_.split("_c")[1].split("_d")[0]), int(name_[-1])
            ops = parse_chp(os.path.join(run, "experiments", "LRB", str(c), "4",
                                         f"{di}.chp"))
        fast = order2_circuit(ops, fast=True)
        for V in fast:
            for k_, v in fast[V].items():
                if k_ in res[name_][V]:
                    maxdiff = max(maxdiff, abs(v - res[name_][V][k_]))
    log(f"  order2_fast vs reference loop on these {len(res)} circuits: max |diff| ="
        f" {maxdiff:.2e}")
    assert maxdiff < 1e-6
    REPORT["G4_order2"] = dict(results=res, memory_increments=inc,
                               fast_vs_reference_max_abs_diff=maxdiff)
    return res


def order2_fast(resp, wts, locid, loc_prob, n_sig, cls_cols):
    """Same quantities as order2_analysis, computed from signature groups.

    Pairs are summed group by group: for every accepted target signature K
    (single-fault signatures with one class, plus K = 0) and every group s,
    the ordered pairs (s, K - s) contribute sum_{a,b} W[s,a] W[K-s,b]
    [a + b != class(K)].  Ordered pairs inside one location (including a
    fault with itself) are subtracted and the result is halved, which gives
    the sum over unordered pairs of faults at distinct locations used by
    order2_analysis.  Checked against order2_analysis in order2_sweep."""
    sig = (resp[:, :n_sig] % D).astype(np.int64)
    ncd = len(cls_cols)
    cl = (resp[:, cls_cols] % D).astype(np.int64)
    ncl = D ** ncd
    cpow = D ** np.arange(ncd, dtype=np.int64)
    cidx = cl @ cpow
    assert n_sig <= 39
    pow3 = D ** np.arange(n_sig, dtype=np.int64)
    key = sig @ pow3
    uk, first, inv = np.unique(key, return_index=True, return_inverse=True)
    inv = inv.reshape(-1)
    G = len(uk)
    W1 = np.zeros((G, ncl))
    np.add.at(W1, (inv, cidx), wts)
    nclass1 = (W1 > 0).sum(axis=1)
    best1 = W1.argmax(axis=1)
    zpos = int(np.searchsorted(uk, 0))
    has_zero = zpos < G and uk[zpos] == 0
    acc1 = nclass1 == 1
    if has_zero:
        acc1[zpos] = True
        best1[zpos] = 0
    gsig = sig[first]
    tot = W1.sum(axis=1)
    dig = np.array([[(c // D ** k) % D for k in range(ncd)] for c in range(ncl)])
    SUB = np.array([[int(((dig[c] - dig[a]) % D) @ cpow) for a in range(ncl)]
                    for c in range(ncl)])
    tkeys = list(uk[acc1])
    tcls = list(best1[acc1])
    tvecs = list(gsig[acc1])
    if not has_zero:
        tkeys.append(0)
        tcls.append(0)
        tvecs.append(np.zeros(n_sig, dtype=np.int64))
    tkeys = np.array(tkeys, dtype=np.int64)
    srt_t = np.argsort(tkeys)
    tkeys = tkeys[srt_t]
    tcls = np.array(tcls, dtype=np.int64)[srt_t]
    tvecs = np.array(tvecs, dtype=np.int64).reshape(-1, n_sig)[srt_t]
    ord_hyb = ord_det = 0.0
    for Kv, kk, cK in zip(tvecs, tkeys, tcls):
        tk = ((Kv[None, :] - gsig) % D) @ pow3
        pos = np.minimum(np.searchsorted(uk, tk), G - 1)
        ok = uk[pos] == tk
        si, ti = np.nonzero(ok)[0], pos[ok]
        same = (W1[si] * W1[ti][:, SUB[cK]]).sum(axis=1)
        val = float((tot[si] * tot[ti] - same).sum())
        ord_hyb += val
        if kk == 0:
            ord_det += val
    srt = np.argsort(locid, kind="stable")
    bounds = np.flatnonzero(np.diff(locid[srt])) + 1
    same_hyb = same_det = 0.0
    for rows in np.split(srt, bounds):
        s_ = sig[rows]
        k2 = ((s_[:, None, :] + s_[None, :, :]) % D) @ pow3
        c2 = ((cl[rows][:, None, :] + cl[rows][None, :, :]) % D) @ cpow
        w2 = wts[rows][:, None] * wts[rows][None, :]
        pos = np.minimum(np.searchsorted(tkeys, k2), len(tkeys) - 1)
        bad = (tkeys[pos] == k2) & (c2 != tcls[pos])
        same_hyb += float(w2[bad].sum())
        same_det += float(w2[bad & (k2 == 0)].sum())
    A2_hyb = (ord_hyb - same_hyb) / 2
    A2_det = (ord_det - same_det) / 2
    undet = float(wts[(key == 0) & (cidx != 0)].sum())
    det_rej1 = float(wts[key != 0].sum())
    hyb_rej1 = float(wts[~acc1[inv]].sum())
    hyb_fail1 = float(wts[acc1[inv] & (cidx != best1[inv])].sum())
    T = float(np.sum(loc_prob))
    f1d = np.nonzero((key == 0) & (cidx != 0))[0]
    f1h = np.nonzero(acc1[inv] & (cidx != best1[inv]))[0]
    dep_det = float(np.sum(wts[f1d] * (T - loc_prob[locid[f1d]])))
    dep_hyb = float(np.sum(wts[f1h] * (T - loc_prob[locid[f1h]])))
    return dict(det_rej1=det_rej1, det_fail_given_acc_1=undet,
                det_fail_given_acc_2=A2_det - dep_det + undet * det_rej1,
                hyb_rej1=hyb_rej1, hyb_fail_given_acc_1=hyb_fail1,
                hyb_fail_given_acc_2=A2_hyb - dep_hyb + hyb_fail1 * hyb_rej1,
                pairs_det=A2_det, pairs_hyb=A2_hyb, n_faults=int(len(resp)),
                n_groups=int(G))


def order2_lrb_sweep(run, cliffs, depth_idx=(1, 2, 3)):
    """Revision (critique C18, C19): two-fault coefficients of the baseline
    LRB circuits for every Clifford sequence at depths m = 2, 4, 6, and the
    per-sequence p^2 slope of the hybrid's conditional failure."""
    log("=" * 78)
    log("G4b. Two-fault coefficients for all sequences (group-based routine)")
    depths = [int(t) for t in open(os.path.join(run, "depths.txt")).read()
              .strip().split(",")]
    t0 = time.time()
    m0 = order2_circuit(parse_chp(os.path.join(run, "experiments", "LRB", "0",
                                               "4", "0.chp")), fast=True)
    per = {0: [m0] * len(cliffs)}
    for di in depth_idx:
        per[depths[di]] = []
        for c in cliffs:
            path = os.path.join(run, "experiments", "LRB", str(c), "4", f"{di}.chp")
            per[depths[di]].append(order2_circuit(parse_chp(path), fast=True))
    log(f"  {len(cliffs)} sequences x {len(depth_idx)} depths in"
        f" {time.time() - t0:.1f} s")
    ms = sorted(per)
    summary = {}
    for V in ("V1", "V2"):
        for key in ("det_fail_given_acc_2", "hyb_fail_given_acc_2", "hyb_rej1",
                    "det_rej1"):
            arr = np.array([[per[m][ci][V][key] for m in ms]
                            for ci in range(len(cliffs))])       # (nseq, nm)
            summary[f"{V}_{key}"] = dict(
                depths=ms, mean=arr.mean(axis=0).tolist(),
                std=arr.std(axis=0, ddof=1).tolist(),
                min=arr.min(axis=0).tolist(), max=arr.max(axis=0).tolist())
            mm = np.array(ms, dtype=float)
            selm = mm >= 2
            A_ = np.vstack([np.ones(selm.sum()), mm[selm]]).T
            slopes = [float(np.linalg.lstsq(A_, a[selm], rcond=None)[0][1])
                      for a in arr]
            slopes_02 = ((arr[:, ms.index(2)] - arr[:, 0]) / 2).tolist()
            summary[f"{V}_{key}"]["slope_m2to6_per_seq"] = _spread(slopes)
            summary[f"{V}_{key}"]["slope_m0to2_per_seq"] = _spread(slopes_02)
            coef = np.linalg.lstsq(A_, arr.mean(axis=0)[selm], rcond=None)[0]
            summary[f"{V}_{key}"]["fit_of_mean_m2to6"] = dict(
                intercept=float(coef[0]), slope_per_depth=float(coef[1]))
    for k_, v in summary.items():
        log(f"  {k_:28s} mean by m={v['depths']}: {np.round(v['mean'], 2).tolist()}"
            f" | per-seq range at m=2: [{v['min'][1]:.1f}, {v['max'][1]:.1f}]")
        log(f"  {'':28s} slope (m=2..6 fit, per seq): mean {v['slope_m2to6_per_seq']['mean']:.2f}"
            f" std {v['slope_m2to6_per_seq']['std']:.2f} range [{v['slope_m2to6_per_seq']['min']:.1f},"
            f" {v['slope_m2to6_per_seq']['max']:.1f}] | (m2-m0)/2: mean "
            f"{v['slope_m0to2_per_seq']['mean']:.2f} range [{v['slope_m0to2_per_seq']['min']:.1f},"
            f" {v['slope_m0to2_per_seq']['max']:.1f}]")
    REPORT["G4b_order2_all_sequences"] = dict(
        cliffords=list(cliffs), summary=summary,
        per_sequence={str(m): [{V: {k: d[V][k] for k in ("det_fail_given_acc_2",
                                                         "hyb_fail_given_acc_2",
                                                         "hyb_rej1", "det_rej1")}
                                for V in d} for d in per[m]] for m in ms})
    return per, summary


def fidelity_comparison(run, summary, u=4.95):
    """Revision (critique C18, missing item 8): predicted unif-1 (detection)
    fidelity 1 - 1.5 (u p + c2 p^2), with c2 the 30-sequence mean of the
    detection P(fail|acc) p^2 coefficient, against the baseline fidelity
    and its standard error (std over Cliffords / sqrt(30))."""
    log("=" * 78)
    log("G4c. Predicted versus baseline unif-1 fidelity (detection)")
    probs = [float(t) for t in open(os.path.join(run, "probs.txt")).read()
             .strip().split(",")]
    depths = [int(t) for t in open(os.path.join(run, "depths.txt")).read()
              .strip().split(",")]
    s = summary["V1_det_fail_given_acc_2"]
    out = []
    for pi in (2, 3, 4, 5, 6):
        p = probs[pi]
        b = baseline_csv(run, pi, "unif_check_data", 1)
        for m, c2 in zip(s["depths"], s["mean"]):
            di = depths.index(m)
            F_b = b["Fidelity averages"][di]
            se = b["Fidelity Standard Deviations"][di] / np.sqrt(30)
            pred2 = 1 - 1.5 * (u * p + c2 * p * p)
            pred1 = 1 - 1.5 * u * p
            row = dict(p=p, m=m, baseline=F_b, se=se, pred_order2=pred2,
                       pred_order1=pred1, diff_order2=F_b - pred2,
                       diff_order1=F_b - pred1, c2=c2)
            out.append(row)
            log(f"  p={p:.0e} m={m:2d}: baseline {F_b:.6f} (SE {se:.1e}) | order-2 "
                f"prediction {pred2:.6f} (diff {F_b - pred2:+.2e} = "
                f"{(F_b - pred2) / se:+.1f} SE) | order-1 only {pred1:.6f} (diff "
                f"{F_b - pred1:+.2e})")
    REPORT["G4c_fidelity_comparison"] = out
    return out


def catalog_group_check(worktree):
    """Side check: how many distinct single-qutrit Clifford unitaries (mod
    global phase) does the 216-string catalog contain?"""
    w = np.exp(2j * np.pi / 3)
    Xm = np.roll(np.eye(3), 1, axis=0)                 # X|j> = |j+1>
    Zm = np.diag([1, w, w * w])
    Fm = np.array([[w ** (j * k) for k in range(3)] for j in range(3)]) / np.sqrt(3)
    # choose H with H X H^dag = Z and H Z H^dag = X^-1 (sdim convention)
    Hm = None
    for cand in (Fm, Fm.conj().T):
        a = cand @ Xm @ cand.conj().T
        b = cand @ Zm @ cand.conj().T
        if np.allclose(a, Zm) and np.allclose(b, Xm.T):
            Hm = cand
    Pm = np.diag([1, 1, w])                            # P X P^dag = XZ
    assert np.allclose(Pm @ Xm @ Pm.conj().T, Xm @ Zm)
    M = {"H": Hm, "P": Pm, "X": Xm, "Z": Zm, "I": np.eye(3)}
    path = os.path.join(worktree, "src", "lrb", "code_definitions.py")
    txt = open(path).read()
    blk = txt[txt.index("single_qutrit_cliffords_exhaustive_with_S_strings = ["):]
    blk = blk[:blk.index("]")]
    strs = [s.replace("S", "P") for s in re.findall(r'"([A-Z]+)"', blk)]
    keys = []
    for s in strs:
        U = np.eye(3, dtype=complex)
        for ch in s:              # descriptor written left to right; the
            U = U @ M[ch]         # rightmost symbol acts first
        k = np.flatnonzero(np.abs(U.ravel()) > 1e-9)[0]
        U = U / (U.ravel()[k] / abs(U.ravel()[k]))
        keys.append(tuple(np.round(U.ravel(), 6)))
    distinct = len(set(keys))

    def norm(U):
        k = np.flatnonzero(np.abs(U.ravel()) > 1e-9)[0]
        U = U / (U.ravel()[k] / abs(U.ravel()[k]))
        return tuple(np.round(U.ravel(), 6)), U

    # closure of <H, P, X, Z> modulo phase
    group = {norm(np.eye(3, dtype=complex))[0]}
    frontier = [np.eye(3, dtype=complex)]
    while frontier:
        new = []
        for U in frontier:
            for g in (Hm, Pm, Xm, Zm):
                k, V = norm(U @ g)
                if k not in group:
                    group.add(k)
                    new.append(V)
        frontier = new
    mult = Counter(Counter(keys).values())
    dup_strings = {s: c for s, c in Counter(strs).items() if c > 1}
    out = dict(n_strings=len(strs), n_distinct_strings=len(set(strs)),
               n_distinct_unitaries_mod_phase=distinct, H_found=Hm is not None,
               group_order_mod_phase=len(group),
               n_group_elements_never_sampled=len(group - set(keys)),
               unitary_multiplicities={str(k): v for k, v in mult.items()},
               duplicated_strings=dup_strings)
    # Revision (critique C21): which elements are never sampled, and do the
    # strings that the critique proposes as intended entries produce them?
    missing = group - set(keys)
    cand = ["XZHSS", "XZSSH", "XXZ", "XXZHSS", "XXZSSH", "ZZX", "ZZXHSS",
            "ZZXSSH", "XXZZHSS", "XXZZSSH"]
    cand_keys = []
    for s in cand:
        U = np.eye(3, dtype=complex)
        for ch in s.replace("S", "P"):
            U = U @ M[ch]
        cand_keys.append(norm(U)[0])
    out["n_missing"] = len(missing)
    out["critique_candidate_strings"] = cand
    out["candidates_are_exactly_the_missing_elements"] = (
        set(cand_keys) == missing and len(set(cand_keys)) == len(cand))
    log("catalog side check:", out)
    REPORT["catalog_check"] = out
    return out


def catalog_twirl_check(worktree):
    """Revision (critique C23, missing item 6).  Logical symplectic part of
    each catalog Clifford, computed from the token maps of section F
    (H: (a,b) -> (-b,a); P: (a,b) -> (a,a+b); X, Z: identity) in execution
    order (the rightmost symbol of a string acts first).  A logical residual
    v created in the check round after forward Clifford j reaches the
    readout conjugated by (C_j ... C_1)^-1 (nested sequences, inverse
    block), so it flips the X readout iff the b component of
    M(C_j ... C_1)^-1 v is nonzero."""
    log("=" * 78)
    log("G5. Catalog: symplectic distribution and readout-flip probabilities")
    path = os.path.join(worktree, "src", "lrb", "code_definitions.py")
    txt = open(path).read()
    blk = txt[txt.index("single_qutrit_cliffords_exhaustive_with_S_strings = ["):]
    blk = blk[:blk.index("]")]
    strs = [s.replace("S", "P") for s in re.findall(r'"([A-Z]+)"', blk)]
    LAM = {"H": np.array([[0, 2], [1, 0]]), "P": np.array([[1, 0], [1, 1]]),
           "X": np.eye(2, dtype=np.int64), "Z": np.eye(2, dtype=np.int64),
           "I": np.eye(2, dtype=np.int64)}
    # consistency with the logical maps computed in section F
    for tok, ent in (("H", "H"), ("P", "P")):
        lam_f = np.array(REPORT["F_gates"][ent]["Lam"])
        assert np.array_equal(lam_f % D, LAM[tok] % D), (tok, lam_f)

    def mkey(Mx):
        return tuple(int(t) for t in (Mx % D).ravel())

    def mat(k):
        return np.array(k, dtype=np.int64).reshape(2, 2)

    dist = Counter()
    for s in strs:
        Mx = np.eye(2, dtype=np.int64)
        for ch in s:                       # product as written: rightmost first
            Mx = (Mx @ LAM[ch]) % D
        dist[mkey(Mx)] += 1
    nz = [(a, b) for a in range(3) for b in range(3) if (a, b) != (0, 0)]

    def inv2(Mx):
        a, b, c, d = Mx.ravel()
        det = int((a * d - b * c) % D)
        dinv = 1 if det == 1 else 2
        return (np.array([[d, -b], [-c, a]]) * dinv) % D

    rows = []
    cur = Counter({mkey(np.eye(2, dtype=np.int64)): 1})
    tot_w = 1
    for j in range(1, 5):
        new = Counter()
        for k1, w1 in cur.items():
            for k2, w2 in dist.items():
                new[mkey(mat(k2) @ mat(k1))] += w1 * w2
        cur, tot_w = new, tot_w * len(strs)
        ent = dict(draws=j)
        for name_, v in (("X_L", (1, 0)), ("Z_L", (0, 1)), ("X_LZ_L", (1, 1)),
                         ("X_LZ_L^-1", (1, 2))):
            img = Counter()
            for k_, w_ in cur.items():
                u = tuple(int(t) for t in (inv2(mat(k_)) @ np.array(v)) % D)
                img[u] += w_
            pf = Fraction(sum(w_ for u, w_ in img.items() if u[1] != 0), tot_w)
            tv = sum(abs(Fraction(img.get(u, 0), tot_w) - Fraction(1, 8))
                     for u in nz) / 2
            ent[name_] = dict(p_flip=str(pf), p_flip_float=float(pf),
                              tv_from_uniform=float(tv))
        rows.append(ent)
        log(f"  {j} draw(s): P(flip) X_L {ent['X_L']['p_flip']} ="
            f" {ent['X_L']['p_flip_float']:.4f}, Z_L {ent['Z_L']['p_flip']} ="
            f" {ent['Z_L']['p_flip_float']:.4f}, X_LZ_L {ent['X_LZ_L']['p_flip']} ="
            f" {ent['X_LZ_L']['p_flip_float']:.4f}, X_LZ_L^-1"
            f" {ent['X_LZ_L^-1']['p_flip_float']:.4f}; max TV distance from uniform"
            f" {max(ent[k]['tv_from_uniform'] for k in ('X_L', 'Z_L', 'X_LZ_L', 'X_LZ_L^-1')):.4f}")
    weights = sorted(dist.values())
    log(f"  SL(2,3) elements hit: {len(dist)} of 24; string counts per element:"
        f" {dict(Counter(weights))} (uniform would be 9 each)")
    REPORT["G5_catalog_twirl"] = dict(n_strings=len(strs),
                                      sl23_weights=dict(Counter(weights)),
                                      flip_probabilities=rows)
    return rows


def logical_level_check():
    """Revision (critique C21): logical action of the physical token circuits
    on the code space, by statevector on the 5 data qutrits (sdim matrices:
    X|j> = |j+1>, Z = diag(1, w, w^2), H with H X H^dag = Z and
    H Z H^dag = X^-1, P = diag(1, 1, w), CNOT|a,b> = |a, a+b>,
    CZ|a,b> = w^(ab)|a,b>).  Logical basis |j_L> is the Z_L = w^j component
    of the encoded |+>_L, so that X_L|j_L> = |j+1_L>."""
    log("=" * 78)
    log("G6. Logical action of the token circuits (5-qutrit statevector)")
    w = np.exp(2j * np.pi / 3)
    Xm = np.roll(np.eye(3), 1, axis=0)
    Zm = np.diag([1, w, w * w])
    Fm = np.array([[w ** (j * k) for k in range(3)] for j in range(3)]) / np.sqrt(3)
    Hm = next(c for c in (Fm, Fm.conj().T)
              if np.allclose(c @ Xm @ c.conj().T, Zm)
              and np.allclose(c @ Zm @ c.conj().T, Xm.T))
    Pm = np.diag([1, 1, w])
    one = {"H": Hm, "H_INV": Hm.conj().T, "P": Pm, "P_INV": Pm.conj().T,
           "X": Xm, "X_INV": Xm.T, "Z": Zm, "Z_INV": Zm.conj(), "I": np.eye(3)}
    n = N
    dim = 3 ** n
    digits = np.array([[(i // 3 ** (n - 1 - q)) % 3 for q in range(n)]
                       for i in range(dim)])

    def idx(dg):
        return int(sum(int(d) * 3 ** (n - 1 - q) for q, d in enumerate(dg)))

    def op1(U, q):
        mats = [np.eye(3)] * n
        mats = [U if k == q else np.eye(3) for k in range(n)]
        out = mats[0]
        for m_ in mats[1:]:
            out = np.kron(out, m_)
        return out

    def op2(g, c, t):
        Mx = np.zeros((dim, dim), dtype=complex)
        for i in range(dim):
            dg = digits[i].copy()
            if g in ("CNOT", "CNOT_INV"):
                dg[t] = (dg[t] + (1 if g == "CNOT" else -1) * dg[c]) % 3
                Mx[idx(dg), i] = 1
            elif g == "SWAP":
                dg[c], dg[t] = dg[t], dg[c]
                Mx[idx(dg), i] = 1
            elif g in ("CZ", "CZ_INV"):
                Mx[i, i] = w ** ((1 if g == "CZ" else -1) * dg[c] * dg[t])
        return Mx

    def circuit(ops):
        U = np.eye(dim, dtype=complex)
        for o in ops:
            G = op1(one[o[0]], o[1]) if len(o) == 2 else op2(o[0], o[1], o[2])
            U = G @ U
        return U

    psi0 = np.zeros(dim, dtype=complex)
    psi0[0] = 1
    plus = circuit([(g, *qs) for g, qs, _ in ENCODER_OPS]) @ psi0
    ZL = op1(Zm, 0) @ op1(Zm, 2)
    XL = op1(Xm, 0) @ op1(Xm, 3)
    stabs = [op1(Zm, 0) @ op1(Zm.conj(), 1) @ op1(Zm.conj(), 3),
             op1(Zm, 1) @ op1(Zm, 2) @ op1(Zm.conj(), 4),
             op1(Xm, 0) @ op1(Xm, 1) @ op1(Xm.T, 2),
             op1(Xm.T, 1) @ op1(Xm, 3) @ op1(Xm.T, 4)]
    expect = [complex(plus.conj() @ S @ plus) for S in stabs + [XL, ZL]]
    basis = []
    for j in range(3):
        Pj = sum(w ** (-j * k) * np.linalg.matrix_power(ZL, k) for k in range(3)) / 3
        v = Pj @ plus
        basis.append(v / np.linalg.norm(v))
    B = np.array(basis).T                                  # (243, 3)
    assert np.allclose(XL @ B[:, 0], B[:, 1]) and np.allclose(XL @ B[:, 2], B[:, 0])
    res = {}
    for tok in ("H", "P", "X", "Z", "H_INV", "P_INV", "X_INV", "Z_INV"):
        U = circuit(logical_gate_ops(tok))
        L = B.conj().T @ U @ B
        target = one[tok]
        k = np.flatnonzero(np.abs(target.ravel()) > 1e-9)[0]
        ph = L.ravel()[k] / target.ravel()[k]
        res[tok] = dict(code_space_preserved=bool(np.allclose(L.conj().T @ L, np.eye(3))),
                        equals_physical_up_to_phase=bool(abs(abs(ph) - 1) < 1e-9
                                                         and np.allclose(L, ph * target)))
    log("  <S_Z1>,<S_Z2>,<S_X1>,<S_X2>,<X_L>,<Z_L> on the encoded state:",
        [complex(round(e.real, 9), round(e.imag, 9)) for e in expect])
    log("  logical action of each token equals the physical single-qutrit gate up to"
        " a global phase:", {k: v["equals_physical_up_to_phase"] and
                             v["code_space_preserved"] for k, v in res.items()})
    REPORT["G6_logical_level"] = dict(
        encoded_expectations=[[e.real, e.imag] for e in expect], tokens=res)
    return res


def generator_semantics_check(val_rows, worktree, depths):
    """Revision (critique C20): check the token sequences recovered from the
    270 baseline files against the catalog and the generator's inverse
    construction: every forward Clifford is a catalog string executed right
    to left; the inverse block is [inverse of each symbol of the string,
    left to right] for the forward Cliffords in reverse order; depth m uses
    the first m Cliffords of the sequence's depth-22 sample."""
    path = os.path.join(worktree, "src", "lrb", "code_definitions.py")
    txt = open(path).read()
    blk = txt[txt.index("single_qutrit_cliffords_exhaustive_with_S_strings = ["):]
    blk = blk[:blk.index("]")]
    cat = {s.replace("S", "P") for s in re.findall(r'"([A-Z]+)"', blk)}
    ok_member = ok_inv = ok_nested = 0
    n = 0
    for c, rows in val_rows.items():
        full = rows[-1]["forward_tokens"]
        for di, r in enumerate(rows):
            fwd, inv = r["forward_tokens"], r["inverse_tokens"]
            n += 1
            ok_member += all("".join(reversed(f)) in cat for f in fwd)
            exp_inv = [INV1[t] for f in reversed(fwd) for t in reversed(f)]
            ok_inv += (list(inv) == exp_inv)
            ok_nested += (fwd == full[:depths[di]])
    out = dict(n_files=n, forward_cliffords_in_catalog=ok_member,
               inverse_block_matches_generator=ok_inv,
               nested_prefix_of_depth22_sample=ok_nested)
    log("generator semantics (270 files):", out)
    REPORT["G0_generator_semantics"] = out
    return out


def final_round_schedule_variant():
    """Revision (critique C13, missing item 7): the undetected floor of the
    final round when ancillas 7 and 8 are measured in one layer after both
    X-check blocks (same SI1000 rules: 2p idle on every other wire per
    measurement layer, p and 5p on each measured ancilla)."""
    log("=" * 78)
    log("G7. Final-round schedule variant (M7 and M8 in one layer)")
    ops = []
    for g, a, b in (("CNOT", 0, 5), ("CNOT_INV", 1, 5), ("CNOT_INV", 3, 5)):
        ops += gate_with_noise(g, (a, b))
    ops += meas_layer(5)
    for g, a, b in (("CNOT", 1, 6), ("CNOT", 2, 6), ("CNOT_INV", 4, 6)):
        ops += gate_with_noise(g, (a, b))
    ops += meas_layer(6)
    for anc, cn in ((7, (("CNOT", 7, 0), ("CNOT", 7, 1), ("CNOT_INV", 7, 2))),
                    (8, (("CNOT", 8, 3), ("CNOT_INV", 8, 1), ("CNOT_INV", 8, 4)))):
        ops += gate_with_noise("H", (anc,))
        for g, a, b in cn:
            ops += gate_with_noise(g, (a, b))
        ops += gate_with_noise("H_INV", (anc,))
    ops += [n1(w_, 2.0) for w_ in range(NQ) if w_ not in (7, 8)]
    ops += [n1(7, 1.0), n1(7, 5.0, "f"), n1(8, 1.0), n1(8, 5.0, "f"),
            ("M", (7,), {}), ("M", (8,), {})]
    std = lrb_circuit_ops([], [])
    ann = [o for o in std if o[0] in ("DETECTOR", "LOGICAL_OBSERVABLE")]
    var = list(ENCODER_OPS) + ops + list(TERMINAL_OPS) + ann
    r_std = analyse_circuit(std, want_breakdown=False)["V1_as_run"]
    r_var = analyse_circuit(var, want_breakdown=False)["V1_as_run"]
    for name_, r in (("as run", r_std), ("M7+M8 merged", r_var)):
        log(f"  final round alone ({name_}): undetected {r['undetected_fail']:.3f} p,"
            f" detection {r['det_rej']:.3f} p, MLD {r['mld_fail']:.3f} p,"
            f" hybrid rejection {r['hyb_rej']:.3f} p")
    REPORT["G7_final_round_schedule"] = dict(as_run=r_std, merged_M7_M8=r_var)
    return r_std, r_var


def pheno_F2_kappa():
    """Revision (critique C12, missing item 3): decomposition of the F2
    (ordinary readout, class b) hybrid q^2 failure into data-data,
    data-measurement and measurement-measurement pairs.  The acceptance set
    does not depend on kappa > 0, so the coefficient is exactly
    c_dd + c_dm kappa + c_mm kappa^2."""
    log("=" * 78)
    log("P2. F2 hybrid q^2 failure by pair type")
    out = {}
    for R in (1, 2, 3):
        res_k = {}
        for kappa in (Fraction(1, 4), Fraction(1), Fraction(4)):
            faults, L = pheno_faults(R, "none", False, "obs", kappa)
            zero = tuple([0] * L)
            T1 = defaultdict(lambda: defaultdict(Fraction))
            for loc, sig, c, w, _ in faults:
                T1[sig][c] += w
            tally = defaultdict(Fraction)
            mm_examples = Counter()
            for i in range(len(faults)):
                for j in range(i + 1, len(faults)):
                    if faults[i][0] == faults[j][0]:
                        continue
                    sig = tuple((np.array(faults[i][1]) + np.array(faults[j][1])) % 3)
                    if sig == zero or sig not in T1:
                        cls_ok = (sig == zero)
                        if not cls_ok:
                            continue          # no single-fault explanation:
                        best = 0              # cannot fail at second order
                    else:
                        t1 = T1[sig]
                        if sum(1 for h in t1 if t1[h] > 0) != 1:
                            continue          # rejected by the strict hybrid
                        best = next(h for h in t1 if t1[h] > 0)
                    c = (faults[i][2] + faults[j][2]) % 3
                    if c == best:
                        continue
                    typ = "".join(sorted(faults[i][0][0] + faults[j][0][0]))
                    tally[typ] += faults[i][3] * faults[j][3]
                    if typ == "mm":
                        mm_examples[(faults[i][4].split(":")[1].split("+")[0],
                                     faults[j][4].split(":")[1].split("+")[0],
                                     faults[i][0][1], faults[j][0][1])] += 1
            ref, _ = pheno_analyse(R, "none", False, "obs", kappa, 3)
            total = sum(tally.values())
            res_k[str(kappa)] = dict(dd=str(tally["dd"]), dm=str(tally["dm"]),
                                     mm=str(tally["mm"]), total=str(total),
                                     reference=str(ref["hyb_fail_acc2"]),
                                     mm_pairs=[f"{a}&{b} (rounds {r1},{r2}): {n_}"
                                               for (a, b, r1, r2), n_ in
                                               sorted(mm_examples.items())])
            assert total == ref["hyb_fail_acc2"], (R, kappa, total, ref)
        k1 = res_k["1"]
        log(f"  R={R}: c(kappa) = {k1['dd']} + {k1['dm']} kappa + {k1['mm']} kappa^2"
            f" (checked at kappa = 1/4, 1, 4); measurement-measurement pairs:"
            f" {k1['mm_pairs']}")
        out[R] = res_k
    REPORT["P2_F2_kappa_decomposition"] = out
    return out


# =============================================================================
# main
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--out", default=here)
    ap.add_argument("--run", default="/pscratch/sd/s/sohan100/LRB-Simulations-sdim/"
                    "LRB-experiment-data-slurm/Run-2026-08-04-23-58-49-"
                    "folded_qutrit-single-ancilla-si1000")
    ap.add_argument("--worktree", default=os.path.abspath(
        os.path.join(here, "..", "..", "..", "..", "..")))
    ap.add_argument("--n-cliffords", type=int, default=30)
    ap.add_argument("--dem-check", action="store_true")
    ap.add_argument("--skip-circuits", action="store_true")
    ap.add_argument("--skip-order2", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    check_code()
    rows, w1rows, Ncnt = classify_syndromes()
    logical_reps()
    code_capacity(["depolarizing", "pure_X", "pure_Z", "pure_X_or_Z",
                   "mixed_only", "centre_only", "corners_only"]
                  + [f"qutrit_{j}_only" for j in range(N)]
                  + ["unequal_pair_X0_X3inv", "single_pure_X0"])
    gate_analysis(w1rows)
    phenomenological()
    # exact finite-q check of the phenomenological leading-order numbers
    ex = {}
    for R in (1, 2):
        for (q, qm) in ((1e-4, 1e-4), (1e-4, 2.5e-5)):
            for final, cm in (("full", "full"), ("none", "obs")):
                r = pheno_exact(q, qm, R, final, cm)
                ex[f"R={R}|q={q}|qm={qm}|{final}|{cm}"] = dict(
                    mld_fail_over_q=r["mld_fail"] / q,
                    det_rej_over_q=(1 - r["det_acc"]) / q,
                    det_fail_given_acc_over_q2=r["det_fail"] / r["det_acc"] / q ** 2)
    log("exact phenomenological checks:", json.dumps(ex, indent=1))
    REPORT["P_exact_checks"] = ex
    pheno_F2_kappa()
    if not args.skip_circuits:
        if os.path.isdir(args.run):
            val = []
            val_rows = defaultdict(list)
            for c in range(30):
                for d in range(9):
                    v = validate_generator(os.path.join(
                        args.run, "experiments", "LRB", str(c), "4", f"{d}.chp"))
                    val.append(v["ok"])
                    val_rows[c].append(v)
            REPORT["G0_generator_validation"] = dict(n=len(val), n_ok=sum(val))
            log("generator reproduces baseline circuit files op-by-op:",
                sum(val), "/", len(val))
            depths = [int(t) for t in open(os.path.join(args.run, "depths.txt"))
                      .read().strip().split(",")]
            generator_semantics_check(val_rows, args.worktree, depths)
        convention_check()
        cycle_table()
        REPORT["G1_gate_noise_per_token"] = gate_noise_per_token()
        REPORT["G1_catalog_token_stats"] = catalog_token_stats(args.worktree)
        log("gate noise per logical token:", REPORT["G1_gate_noise_per_token"])
        log("catalog token stats:", REPORT["G1_catalog_token_stats"])
        catalog_group_check(args.worktree)
        catalog_twirl_check(args.worktree)
        logical_level_check()
        memory_windows(5)
        final_round_schedule_variant()
        if os.path.isdir(args.run):
            lrb_circuits(args.run, list(range(args.n_cliffords)), list(range(9)))
        if not args.skip_order2:
            order2_sweep(args.run)
            if os.path.isdir(args.run):
                _, summ = order2_lrb_sweep(args.run, list(range(args.n_cliffords)))
                fidelity_comparison(args.run, summ)
        if args.dem_check:
            src = os.path.join(args.worktree, "src")
            dc = [dem_crosscheck(os.path.join(args.run, "experiments", sub, "0",
                                              "4", f"{d}.chp"), src)
                  for sub, d in (("LRB", 0), ("LRB", 1), ("LRB", 3),
                                 ("LRB_const0", 1))]
            REPORT["G_dem_crosscheck"] = dc
            log("DEM cross-check:", dc)
    out = os.path.join(args.out, "A1_results.json")
    with open(out, "w") as fh:
        json.dump(REPORT, fh, indent=1, default=lambda o: o.tolist()
                  if isinstance(o, np.ndarray) else str(o))
    log(f"wrote {out}; total time {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
