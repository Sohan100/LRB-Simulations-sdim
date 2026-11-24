"""Adversarial verification of work package G (distance-3 [[13,1,3]] code).

Independent of G's construction: the code, its fold symmetry and the
fold-transversal gates are rebuilt here from the edge picture of the
unrotated planar surface code (data on edges, X checks on vertices, Z checks
on faces), with my own qubit numbering. The LRB circuits are rebuilt with my
own builder and my own SI1000 noise insertion (``MyLRB``). G's modules
(``lrb_qubit.d3_code``, ``lrb_qubit.d3_circuits``) are imported only inside
functions whose name starts with ``compare_`` or ``g_``, to compare with or to
run G's decoders on circuits that are first shown to be identical to mine.
The lead's ``lrb_qubit.code`` supplies the 24-element Clifford catalog and its
gate-token convention, and ``lrb_qubit.experiment.clifford_sequences`` the
main grid's sequences (both are part of the protocol, not of G's work).

Subcommands (outputs go to ``--out``, default ``results/G-verify``):

``code``       claims 1-2: distance, fold symmetry, fold-transversal H and S
               (Stim tableaus, all 32 S/S-dagger patterns), d = 2..5; then the
               comparison with G's labelling and gate lists.
``builder``    claim 4: G's generic builder vs the lead's builder at d = 2.
``circuits``   my LRB builder vs G's: identical detector error models.
``distance``   claim 5: Stim's search on depth-2 circuits (no truncation),
               plus an exhaustive weight <= 2 check on many circuits.
``orders``     claim 6: all 576 uniform CNOT orders, exhaustive weight <= 2
               check, on several sequences (``--seq-set``).
``matching``   claim 7: PyMatching single-fault census on circuits with S.
``bposd``      claim 8: plain ldpc BP-OSD on single-fault syndromes.
``firstorder`` claim 9: own FlipSimulator census of the d = 3 parallel
               circuit, decoded by BP-OSD (E1 should be 0); d = 2 E1 slopes.
``e2``         exact second-order coefficient of the optimal decoder (d = 3).
``sample``     one sampled BP-OSD point (G's decoder on my circuits).
``const0``     exact no-check (const0) MLD error, p* against physical RB.
``lrbd``       LRB-D unif=1 at d = 3 from raw measurement records.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_v) != "1" for _v in _THREAD_VARS):
    for _v in _THREAD_VARS:
        os.environ[_v] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import stim  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import code as LEAD  # noqa: E402  (Clifford catalog only)

OUT_DEFAULT = ROOT / "studies/correction-decoders/qubit/results/G-verify"
SEQ_SEED = 20261028
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
S_AND_H = ["P", "H", "HP", "PH", "HPH", "P", "XZ", "H"]


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, default=_json_default))


def _json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    raise TypeError(type(x))


# ======================================================================
# 1. My construction of the unrotated planar code (edge picture)
# ======================================================================
class EdgeCode:
    """Distance-d unrotated planar surface code, data on edges.

    Vertices v(i, j), i = 0..d-1 (rows), j = 0..d-2, carry the X checks
    (stars). Faces f(i, k), i = 0..d-2, k = 0..d-1, carry the Z checks
    (plaquettes). Horizontal edges h(i, k), k = 0..d-1, where h(i, 0) and
    h(i, d-1) dangle off the left and right (rough) boundaries; vertical
    edges u(i, j) join v(i, j) and v(i+1, j). Top and bottom are smooth.

    X_L = X on the left dangling column {h(i, 0)}; Z_L = Z on row 0
    {h(0, k)}. My numbering: vertical edges first (row-major), then
    horizontal edges (row-major), so it differs from G's on purpose.

    Coordinates (only used for the CNOT compass order and at the very end to
    compare with G): h(i,k) -> (2i, 2k), u(i,j) -> (2i+1, 2j+1),
    v(i,j) -> (2i, 2j+1), f(i,k) -> (2i+1, 2k).
    """

    def __init__(self, d: int):
        self.d = d
        names = ([("u", i, j) for i in range(d - 1) for j in range(d - 1)]
                 + [("h", i, k) for i in range(d) for k in range(d)])
        self.names = names
        self.index = {n: q for q, n in enumerate(names)}
        self.n = len(names)
        self.pos = {}
        for q, (t, a, b) in enumerate(names):
            self.pos[q] = (2 * a, 2 * b) if t == "h" else (2 * a + 1, 2 * b + 1)
        h = lambda i, k: self.index[("h", i, k)]  # noqa: E731
        u = lambda i, j: self.index[("u", i, j)]  # noqa: E731
        self.x_checks = []   # (name, position, support set)
        for i in range(d):
            for j in range(d - 1):
                sup = {h(i, j), h(i, j + 1)}
                if i >= 1:
                    sup.add(u(i - 1, j))
                if i <= d - 2:
                    sup.add(u(i, j))
                self.x_checks.append((("v", i, j), (2 * i, 2 * j + 1),
                                      frozenset(sup)))
        self.z_checks = []
        for i in range(d - 1):
            for k in range(d):
                sup = {h(i, k), h(i + 1, k)}
                if k >= 1:
                    sup.add(u(i, k - 1))
                if k <= d - 2:
                    sup.add(u(i, k))
                self.z_checks.append((("f", i, k), (2 * i + 1, 2 * k),
                                      frozenset(sup)))
        self.x_logical = frozenset(h(i, 0) for i in range(d))
        self.z_logical = frozenset(h(0, k) for k in range(d))
        self.at = {self.pos[q]: q for q in range(self.n)}

    # Pauli helpers -----------------------------------------------------
    def pauli(self, kind: str, support) -> stim.PauliString:
        p = stim.PauliString(self.n)
        for q in support:
            p[q] = kind
        return p

    def generators(self) -> list[stim.PauliString]:
        return ([self.pauli("Z", s) for _, _, s in self.z_checks]
                + [self.pauli("X", s) for _, _, s in self.x_checks])

    def logical(self, which: str) -> stim.PauliString:
        x = self.pauli("X", self.x_logical)
        z = self.pauli("Z", self.z_logical)
        return {"X": x, "Z": z, "Y": x * z * 1j}[which]


def symplectic(p: stim.PauliString) -> np.ndarray:
    x, z = p.to_numpy()
    return np.concatenate([x, z]).astype(np.uint8)


class GF2Span:
    """Membership and coefficients in the GF(2) row span of generators."""

    def __init__(self, rows: np.ndarray):
        rows = np.asarray(rows, dtype=np.uint8) % 2
        self.k, self.n = rows.shape
        # Row-reduce [rows | I] to get, for every reduced row, the subset of
        # original rows that produces it.
        aug = np.concatenate([rows, np.eye(self.k, dtype=np.uint8)], axis=1)
        pivots = []
        r = 0
        for c in range(self.n):
            cand = [i for i in range(r, self.k) if aug[i, c]]
            if not cand:
                continue
            aug[[r, cand[0]]] = aug[[cand[0], r]]
            for i in range(self.k):
                if i != r and aug[i, c]:
                    aug[i] ^= aug[r]
            pivots.append(c)
            r += 1
            if r == self.k:
                break
        self.rank = r
        self.reduced = aug[:r, :self.n]
        self.combo = aug[:r, self.n:]
        self.pivots = pivots

    def solve(self, v: np.ndarray):
        """Coefficient vector c (over the original rows) with c.rows = v."""
        v = np.asarray(v, dtype=np.uint8).copy() % 2
        coeff = np.zeros(self.k, dtype=np.uint8)
        for i, c in enumerate(self.pivots):
            if v[c]:
                v ^= self.reduced[i]
                coeff ^= self.combo[i]
        if v.any():
            return None
        return coeff


class StabilizerGroup:
    """Signed membership tests for a code with +1 generators."""

    def __init__(self, gens: list[stim.PauliString],
                 logicals: dict[str, stim.PauliString]):
        self.gens = gens
        self.span = GF2Span(np.array([symplectic(g) for g in gens]))
        if self.span.rank != len(gens):
            raise ValueError("generators are dependent")
        self.logicals = logicals
        self.n = len(gens[0])

    def product(self, coeff) -> stim.PauliString:
        out = stim.PauliString(self.n)
        for s in np.flatnonzero(coeff):
            out *= self.gens[s]
        return out

    def signed_member(self, p: stim.PauliString):
        """+1 / -1 if +-p is a product of generators, else None."""
        c = self.span.solve(symplectic(p))
        if c is None:
            return None
        prod = self.product(c)
        if p == prod:
            return +1
        if p == -prod:
            return -1
        return None  # imaginary sign: not Hermitian-compatible

    def logical_class(self, p: stim.PauliString):
        """(name, sign) with p = sign * L * stabilizer, or None."""
        for name in ("I", "X", "Y", "Z"):
            L = (stim.PauliString(self.n) if name == "I"
                 else self.logicals[name])
            r = L * p
            c = self.span.solve(symplectic(r))
            if c is None:
                continue
            prod = self.product(c)
            if r == prod:
                return name, +1
            if r == -prod:
                return name, -1
            return name, None
        return None


def code_distance_bruteforce(code: EdgeCode) -> dict:
    """CSS distances by enumerating every X-type and Z-type string."""
    n = code.n
    hz = np.zeros((len(code.z_checks), n), np.uint8)
    hx = np.zeros((len(code.x_checks), n), np.uint8)
    for i, (_, _, s) in enumerate(code.z_checks):
        hz[i, list(s)] = 1
    for i, (_, _, s) in enumerate(code.x_checks):
        hx[i, list(s)] = 1
    span_x = GF2Span(hx)
    span_z = GF2Span(hz)
    best = {"X": n + 1, "Z": n + 1}
    count = {"X": 0, "Z": 0}
    for m in range(1, 2 ** n):
        v = np.array([(m >> q) & 1 for q in range(n)], np.uint8)
        w = int(v.sum())
        # X-type logical: commutes with Z checks, not an X stabilizer.
        if not (hz @ v % 2).any() and span_x.solve(v) is None:
            count["X"] += 1 if w == best["X"] else 0
            if w < best["X"]:
                best["X"], count["X"] = w, 1
        if not (hx @ v % 2).any() and span_z.solve(v) is None:
            count["Z"] += 1 if w == best["Z"] else 0
            if w < best["Z"]:
                best["Z"], count["Z"] = w, 1
    return {"d_X": best["X"], "d_Z": best["Z"],
            "num_min_weight_X_logicals": count["X"],
            "num_min_weight_Z_logicals": count["Z"],
            "rank_hx": span_x.rank, "rank_hz": span_z.rank,
            "k": n - span_x.rank - span_z.rank}


def low_weight_pauli_check(code: EdgeCode, group: StabilizerGroup,
                           max_weight: int = 2) -> dict:
    """Every Pauli of weight <= max_weight: detected, or a stabilizer."""
    gens = group.gens
    undetected_nontrivial = []
    total = 0
    for w in range(1, max_weight + 1):
        for qs in itertools.combinations(range(code.n), w):
            for ps in itertools.product("XYZ", repeat=w):
                total += 1
                p = stim.PauliString(code.n)
                for q, ch in zip(qs, ps):
                    p[q] = ch
                if all(p.commutes(g) for g in gens):
                    cls = group.logical_class(p)
                    if cls is None or cls[0] != "I":
                        undetected_nontrivial.append(str(p))
    return {"paulis_checked": total,
            "undetected_nontrivial": undetected_nontrivial}


def find_dualities(code: EdgeCode) -> list[dict]:
    """All qubit permutations mapping X-check supports onto Z-check supports
    and Z-check supports onto X-check supports (as set families)."""
    xs = [s for _, _, s in code.x_checks]
    zs = [s for _, _, s in code.z_checks]
    zset, xset = set(zs), set(xs)
    inc_z = {}
    for q in range(code.n):
        key = frozenset(i for i, s in enumerate(zs) if q in s)
        inc_z.setdefault(key, []).append(q)
    out = []
    for beta in itertools.permutations(range(len(zs))):
        pi = {}
        ok = True
        for q in range(code.n):
            need = frozenset(beta[i] for i, s in enumerate(xs) if q in s)
            cands = inc_z.get(need, [])
            if len(cands) != 1:
                ok = False
                break
            pi[q] = cands[0]
        if not ok or len(set(pi.values())) != code.n:
            continue
        if {frozenset(pi[q] for q in s) for s in xs} != zset:
            continue
        if {frozenset(pi[q] for q in s) for s in zs} != xset:
            continue
        out.append(pi)
    result = []
    for pi in out:
        involution = all(pi[pi[q]] == q for q in range(code.n))
        result.append({
            "perm": pi, "involution": involution,
            "maps_XL_support_to_ZL_support":
                frozenset(pi[q] for q in code.x_logical) == code.z_logical,
            "fixed": sorted(q for q in range(code.n) if pi[q] == q)})
    return result


def tableau_of(n: int, ops) -> stim.Tableau:
    c = stim.Circuit()
    c.append("I", [n - 1])
    for g, t in ops:
        c.append(g, list(t))
    return stim.Tableau.from_circuit(c)


def gate_action(group: StabilizerGroup, tab: stim.Tableau) -> dict:
    stab_signs = [group.signed_member(tab(g)) for g in group.gens]
    return {"stabilizer_signs": stab_signs,
            "all_stabilizers_plus": all(s == +1 for s in stab_signs),
            "X_L": group.logical_class(tab(group.logicals["X"])),
            "Z_L": group.logical_class(tab(group.logicals["Z"]))}


def fold_gates(fold: dict, n: int, diag_pattern: dict):
    """H = SWAP on mirror pairs then H on all; S = S/S_DAG on fixed
    qubits (diag_pattern) and CZ on mirror pairs."""
    pairs = sorted({tuple(sorted((q, fold[q]))) for q in range(n)
                    if fold[q] != q})
    h_ops = [("SWAP", pr) for pr in pairs] + [("H", (q,)) for q in range(n)]
    s_ops = ([(diag_pattern[q], (q,)) for q in sorted(diag_pattern)]
             + [("CZ", pr) for pr in pairs])
    return pairs, h_ops, s_ops


def analyse_code(d: int) -> dict:
    code = EdgeCode(d)
    group = StabilizerGroup(code.generators(),
                            {k: code.logical(k) for k in "XYZ"})
    rep = {"d": d, "n": code.n, "num_x_checks": len(code.x_checks),
           "num_z_checks": len(code.z_checks)}
    gens = group.gens
    rep["generators_commute"] = all(a.commutes(b) for a, b in
                                    itertools.combinations(gens, 2))
    rep["logicals_commute_with_generators"] = all(
        g.commutes(code.logical(k)) for g in gens for k in "XZ")
    rep["XL_ZL_anticommute"] = not code.logical("X").commutes(code.logical("Z"))
    rep["weights"] = {"x_checks": sorted(len(s) for _, _, s in code.x_checks),
                      "z_checks": sorted(len(s) for _, _, s in code.z_checks),
                      "X_L": len(code.x_logical), "Z_L": len(code.z_logical)}
    if code.n <= 16:
        rep["css_distance"] = code_distance_bruteforce(code)
    if code.n <= 13:
        rep["weight_le_2_paulis"] = low_weight_pauli_check(code, group, 2)
    duals = find_dualities(code)
    rep["num_dualities"] = len(duals)
    rep["dualities"] = [{"involution": x["involution"],
                         "maps_XL_to_ZL": x["maps_XL_support_to_ZL_support"],
                         "num_fixed": len(x["fixed"])} for x in duals]
    chosen = [x for x in duals if x["involution"]
              and x["maps_XL_support_to_ZL_support"]]
    rep["num_fold_involutions_XL_to_ZL"] = len(chosen)
    if not chosen:
        return rep
    fold = chosen[0]["perm"]
    fixed = chosen[0]["fixed"]
    rep["fold_fixed_qubits"] = [
        {"q": q, "name": code.names[q], "pos": code.pos[q]} for q in fixed]
    rep["fold_is_transpose"] = all(
        code.pos[fold[q]] == code.pos[q][::-1] for q in range(code.n))
    # Logical H.
    pairs, h_ops, _ = fold_gates(fold, code.n, {})
    rep["fold_pairs"] = [[code.names[a], code.names[b]] for a, b in pairs]
    rep["H"] = gate_action(group, tableau_of(code.n, h_ops))
    # Logical S: every S / S_DAG pattern on the fixed qubits.
    patterns = []
    for bits in itertools.product((0, 1), repeat=len(fixed)):
        pat = {q: ("S" if b == 0 else "S_DAG") for q, b in zip(fixed, bits)}
        _, _, s_ops = fold_gates(fold, code.n, pat)
        act = gate_action(group, tableau_of(code.n, s_ops))
        patterns.append({
            "pattern": {f"{code.pos[q]}": g for q, g in pat.items()},
            "all_stabilizers_plus": act["all_stabilizers_plus"],
            "num_minus_or_bad": sum(s != +1 for s in act["stabilizer_signs"]),
            "X_L": act["X_L"], "Z_L": act["Z_L"]})
    rep["S_patterns_valid"] = [x for x in patterns if x["all_stabilizers_plus"]]
    rep["S_patterns_total"] = len(patterns)
    # The alternating pattern along the diagonal: S at even (i, i).
    alt = {q: ("S" if code.pos[q][0] % 4 == 0 else "S_DAG") for q in fixed}
    rep["alternating_pattern"] = {f"{code.pos[q]}": g for q, g in alt.items()}
    _, _, s_ops = fold_gates(fold, code.n, alt)
    rep["S_alternating"] = gate_action(group, tableau_of(code.n, s_ops))
    rep["_objects"] = (code, group, fold, h_ops, s_ops)
    return rep


def compare_with_g_labels(code: EdgeCode, group: StabilizerGroup, h_ops,
                          s_ops) -> dict:
    """Map my qubits to G's labels (my (r, c) -> G's site (c, r)) and compare
    supports, logicals and gate tableaus (G's labelling used only here)."""
    from lrb_qubit import d3_code as D
    g = D.CODE13
    perm = {q: g.labels[(code.pos[q][1], code.pos[q][0])]
            for q in range(code.n)}
    rep = {"label_map_my_to_G": {code.names[q].__repr__(): perm[q]
                                 for q in range(code.n)}}
    g_x = {frozenset(g.check_supports[s]) for s in g.x_stabilizers}
    g_z = {frozenset(g.check_supports[s]) for s in g.z_stabilizers}
    my_x = {frozenset(perm[q] for q in s) for _, _, s in code.x_checks}
    my_z = {frozenset(perm[q] for q in s) for _, _, s in code.z_checks}
    rep["x_checks_equal"] = my_x == g_x
    rep["z_checks_equal"] = my_z == g_z
    rep["X_L_equal"] = {perm[q] for q in code.x_logical} == set(
        g.logical_x_support)
    rep["Z_L_equal"] = {perm[q] for q in code.z_logical} == set(
        g.logical_z_support)

    def relabel(ops):
        return [(gate, tuple(perm[t] for t in tg)) for gate, tg in ops]

    for name, mine, token in (("H", h_ops, "H"), ("S", s_ops, "P")):
        t_mine = tableau_of(13, relabel(mine))
        t_g = tableau_of(13, g.logical_gate_operations(token))
        rep[f"{name}_tableau_equal_to_G"] = t_mine == t_g
        rep[f"{name}_ops_equal_as_sets"] = (
            sorted(relabel(mine)) == sorted(g.logical_gate_operations(token)))
    # G's own tableau checks redone with my signed group (relabelled).
    inv = {v: k for k, v in perm.items()}
    gens_g = [stim.PauliString(s) for s in g.stabilizer_paulis]
    gens_mine_in_g = []
    for p in group.gens:
        out = stim.PauliString(13)
        for q in range(13):
            out[perm[q]] = p[q]
        gens_mine_in_g.append(out)
    span = GF2Span(np.array([symplectic(x) for x in gens_mine_in_g]))
    rep["G_generators_in_my_group"] = all(
        span.solve(symplectic(x)) is not None for x in gens_g)
    rep["G_diagonal_gates"] = [[gate, q, list(g.site_of[q])]
                               for gate, q in g.diagonal_gates]
    rep["G_fold_pairs"] = [list(x) for x in g.fold_pairs]
    del inv
    return rep


def clifford_action_check(code, group, h_ops, s_ops) -> dict:
    """All 24 descriptors (lead's catalog) act like the 1-qubit Clifford,
    and their mirror inverses invert them; built from MY H and S."""
    n = code.n
    s_dag = [("S_DAG" if g == "S" else "S" if g == "S_DAG" else g, t)
             for g, t in reversed(s_ops)]
    token_ops = {
        "I": [("I", (0,))], "H": h_ops, "P": s_ops,
        "X": [("X", (q,)) for q in sorted(code.x_logical)],
        "Z": [("Z", (q,)) for q in sorted(code.z_logical)]}
    inverse_ops = {"I": token_ops["I"], "X": token_ops["X"],
                   "Z": token_ops["Z"], "P": s_dag,
                   "H": list(reversed(h_ops))}
    single = {"I": "I", "H": "H", "P": "S", "X": "X", "Z": "Z"}
    bad = []
    for desc in LEAD.SINGLE_QUBIT_CLIFFORDS:
        toks = LEAD.clifford_gate_sequence(desc)
        ops = [o for t in toks for o in token_ops[t]]
        inv = [o for t in desc for o in inverse_ops[t]]
        if tableau_of(n, ops + inv) != stim.Tableau(n):
            bad.append((desc, "inverse"))
        one = stim.Circuit()
        one.append("I", [0])
        for t in toks:
            one.append(single[t], [0])
        t1 = stim.Tableau.from_circuit(one)
        tab = tableau_of(n, ops)
        for which in "XZ":
            exp = t1(stim.PauliString(which))
            want = (str(exp)[1:], int(exp.sign.real))
            got = group.logical_class(tab(group.logicals[which]))
            if got != want:
                bad.append((desc, which, got, want))
        if not all(group.signed_member(tab(g)) == 1 for g in group.gens):
            bad.append((desc, "stabilizer sign"))
    return {"descriptors": len(LEAD.SINGLE_QUBIT_CLIFFORDS), "failures": bad}


def cmd_code(args) -> int:
    out = Path(args.out)
    report = {}
    for d in (2, 3, 4, 5):
        rep = analyse_code(d)
        objs = rep.pop("_objects", None)
        if d == 3 and objs is not None:
            code, group, fold, h_ops, s_ops = objs
            rep["compare_with_G"] = compare_with_g_labels(code, group, h_ops,
                                                          s_ops)
            rep["clifford_catalog"] = clifford_action_check(code, group,
                                                            h_ops, s_ops)
        if d == 2 and objs is not None:
            rep["compare_with_lead_512"] = compare_d2_with_lead(*objs)
        report[f"d{d}"] = rep
        print(json.dumps({k: v for k, v in rep.items()
                          if k not in ("dualities",)}, default=_json_default,
                         indent=None)[:3000], flush=True)
    dump(out / "code.json", report)
    return 0


def compare_d2_with_lead(code, group, fold, h_ops, s_ops) -> dict:
    """Find the embeddings of the lead's [[5,1,2]] into my d=2 code that
    preserve check types, supports and logical supports; compare S."""
    lead_z = [frozenset(q for q, ch in enumerate(s) if ch != "_")
              for s in LEAD.STABILIZER_PAULIS[:2]]
    lead_x = [frozenset(q for q, ch in enumerate(s) if ch != "_")
              for s in LEAD.STABILIZER_PAULIS[2:]]
    lx = frozenset(q for q, ch in enumerate(LEAD.LOGICAL_X) if ch != "_")
    lz = frozenset(q for q, ch in enumerate(LEAD.LOGICAL_Z) if ch != "_")
    my_x = {s for _, _, s in code.x_checks}
    my_z = {s for _, _, s in code.z_checks}
    found = []
    for perm in itertools.permutations(range(5)):
        phi = dict(enumerate(perm))  # lead qubit -> my qubit
        if {frozenset(phi[q] for q in s) for s in lead_x} != my_x:
            continue
        if {frozenset(phi[q] for q in s) for s in lead_z} != my_z:
            continue
        if frozenset(phi[q] for q in lx) != code.x_logical:
            continue
        if frozenset(phi[q] for q in lz) != code.z_logical:
            continue
        # Lead's S in my labels vs my S.
        lead_s = [(g, tuple(phi[t] for t in tg))
                  for g, tg in LEAD.logical_gate_operations("P")]
        lead_h = [(g, tuple(phi[t] for t in tg))
                  for g, tg in LEAD.logical_gate_operations("H")]
        # CNOT compass order of the lead's checks in my coordinates.
        orders = {}
        for name, ops in (("Z", LEAD.z_stabilizer_check_operations()),
                          ("X", LEAD.x_stabilizer_check_operations())):
            seqs = []
            cur = []
            for g, tg in ops:
                if g == "CX":
                    data = tg[0] if name == "Z" else tg[1]
                    cur.append(phi[data])
                if g == "M":
                    seqs.append(cur)
                    cur = []
            dirs = []
            for seq in seqs:
                sup = frozenset(seq)
                pos = [c for c in (code.z_checks if name == "Z"
                                   else code.x_checks) if c[2] == sup][0][1]
                step = []
                for q in seq:
                    dr = code.pos[q][0] - pos[0]
                    dc = code.pos[q][1] - pos[1]
                    step.append({(-1, 0): "N", (1, 0): "S", (0, -1): "W",
                                 (0, 1): "E"}[(dr, dc)])
                dirs.append("".join(step))
            orders[name] = dirs
        found.append({
            "lead_to_my": {q: code.names[phi[q]] for q in range(5)},
            "S_tableau_equal": tableau_of(5, lead_s) == tableau_of(5, s_ops),
            "H_tableau_equal": tableau_of(5, lead_h) == tableau_of(5, h_ops),
            "check_cnot_compass_orders_in_my_coords": orders})
    return {"num_embeddings": len(found), "embeddings": found}


# ======================================================================
# main
# ======================================================================
def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in COMMANDS:
        sp = sub.add_parser(name)
        sp.add_argument("--out", default=str(OUT_DEFAULT))
        sp.add_argument("--workers", type=int, default=4)
        sp.add_argument("--p", type=float, default=1e-3)
        sp.add_argument("--depth", type=int, default=22)
        sp.add_argument("--shots", type=int, default=1000)
        sp.add_argument("--num-seq", type=int, default=10)
        sp.add_argument("--seq-set", default="default")
        sp.add_argument("--budget", type=float, default=600.0)
        sp.add_argument("--schedule", default="parallel")
        sp.add_argument("--seed", type=int, default=424242)
        sp.add_argument("--tag", default="")
    args = parser.parse_args()
    return COMMANDS[args.cmd](args)


COMMANDS = {"code": cmd_code}

if __name__ == "__main__":
    raise SystemExit(main())
