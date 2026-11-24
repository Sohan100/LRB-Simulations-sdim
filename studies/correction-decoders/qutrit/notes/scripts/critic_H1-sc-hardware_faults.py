"""Critic check of H1 claims C4, C6, C17, C23, C26 (independent code).

1. Conjugation rules of CSUM, CSUM^-1, CZ, CZ^-1 (sdim conventions
   CNOT|a,b> = |a,a+b>, CZ|a,b> = w^{ab}|a,b>) are DERIVED here from explicit
   9x9 matrices, not typed by hand, and then used for propagation.
2. C6: recount single two-qutrit-gate faults in the three CZ(2,3) gadgets and
   split "detected" into "data syndrome" versus "only an X on a relay or
   ancilla" (which is detected only if that qutrit is later measured in Z).
3. C26: syndrome ambiguity of single-qutrit data errors, then the same
   question for the first-order data errors that the LRB circuit actually
   produces: all errors supported on {2,3} (N2 after CZ(2,3)/SWAP(2,3) of
   logical P/H) and single errors on 2 or 3 propagated through a CZ(2,3).
4. C4: noiseless sdim circuits with logical H implemented WITHOUT the SWAP
   (frame tracking: X/Z tokens and the check round relabelled, data 2<->3,
   ancillas 5<->7 and 6<->8): every check outcome must be deterministic 0
   and x0+x3 = 0 in every shot. Negative control: same without relabelling.
5. C17/C23 sanity: unreset-relay arithmetic on basis states, and the
   |+>^5 projection argument.

Run:  module load python/3.11; python critic_H1-sc-hardware_faults.py
"""
from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

W = np.exp(2j * np.pi / 3)
REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

# ---------------------------------------------------------------- matrices
X1 = np.roll(np.eye(3), 1, axis=0)          # X|j> = |j+1>
Z1 = np.diag([W ** j for j in range(3)])     # Z|j> = w^j |j>


def weyl1(x, z):
    return np.linalg.matrix_power(X1, x % 3) @ np.linalg.matrix_power(Z1, z % 3)


def weyl2(v):  # v = (x_c, z_c, x_t, z_t)
    return np.kron(weyl1(v[0], v[1]), weyl1(v[2], v[3]))


def gate2(name):
    U = np.zeros((9, 9), complex)
    for a in range(3):
        for b in range(3):
            if name == "CSUM":
                U[3 * a + (a + b) % 3, 3 * a + b] = 1
            elif name == "CSUM_INV":
                U[3 * a + (b - a) % 3, 3 * a + b] = 1
            elif name == "CZ":
                U[3 * a + b, 3 * a + b] = W ** (a * b)
            elif name == "CZ_INV":
                U[3 * a + b, 3 * a + b] = W ** (-a * b)
            elif name == "SWAP":
                U[3 * b + a, 3 * a + b] = 1
    return U


def symplectic_map(name):
    """4x4 matrix M over Z_3 with U W(v) U^+ ~ W(M v)."""
    U = gate2(name)
    basis = [np.eye(4, dtype=int)[i] for i in range(4)]
    cols = []
    for v in basis:
        C = U @ weyl2(v) @ U.conj().T
        hit = None
        for w in itertools.product(range(3), repeat=4):
            if abs(abs(np.trace(weyl2(w).conj().T @ C)) - 9) < 1e-9:
                hit = w
                break
        cols.append(hit)
    M = np.array(cols).T % 3
    # check linearity on all 81 vectors
    for v in itertools.product(range(3), repeat=4):
        C = U @ weyl2(v) @ U.conj().T
        w = tuple(M @ np.array(v) % 3)
        assert abs(abs(np.trace(weyl2(w).conj().T @ C)) - 9) < 1e-9
    return M


SMAP = {g: symplectic_map(g) for g in ("CSUM", "CSUM_INV", "CZ", "CZ_INV",
                                       "SWAP")}

# ---------------------------------------------------------------- code
STAB = {  # (x[0..4], z[0..4])
    "S_Z1": ([0] * 5, [1, -1, 0, -1, 0]),
    "S_Z2": ([0] * 5, [0, 1, 1, 0, -1]),
    "S_X1": ([1, 1, -1, 0, 0], [0] * 5),
    "S_X2": ([0, -1, 0, 1, -1], [0] * 5),
}
XL = ([1, 0, 0, 1, 0], [0] * 5)
ZL = ([0] * 5, [1, 0, 1, 0, 0])


def sp(a, b):
    return int((np.dot(a[0], b[1]) - np.dot(a[1], b[0])) % 3)


def syndrome(e):
    return tuple(sp(e, s) for s in STAB.values())


def logical(e):
    return (sp(e, ZL), sp(e, XL))


def propagate(gadget, k, err, n=11):
    """err on qutrits of gate k, inserted right after gate k."""
    x = np.zeros(n, int)
    z = np.zeros(n, int)
    _, c, t = gadget[k]
    x[c], z[c], x[t], z[t] = err
    for g, c, t in gadget[k + 1:]:
        v = np.array([x[c], z[c], x[t], z[t]])
        w = SMAP[g] @ v % 3
        x[c], z[c], x[t], z[t] = w
    return x, z


GADGETS = {
    "native": [("CZ", 2, 3)],
    "relay": [("CSUM", 3, 9), ("CZ", 2, 9), ("CSUM_INV", 3, 9)],
    "bus": [("CSUM", 2, 7), ("CSUM", 3, 5), ("CSUM", 7, 0), ("CZ", 0, 5),
            ("CSUM_INV", 7, 0), ("CZ_INV", 0, 5), ("CSUM_INV", 2, 7),
            ("CSUM_INV", 3, 5)],
}


def classify(gadget):
    out = {"total": 0, "data_syndrome": 0, "only_anc_X": 0,
           "only_anc_X_and_logical": 0, "trivial": 0, "silent_logical": 0}
    examples = []
    for k in range(len(gadget)):
        for e in itertools.product(range(3), repeat=4):
            if not any(e):
                continue
            x, z = propagate(gadget, k, e)
            d = (x[:5], z[:5])
            syn = syndrome(d)
            ancx = bool(np.any(x[5:] % 3))
            lg = logical(d)
            out["total"] += 1
            if any(syn):
                out["data_syndrome"] += 1
            elif ancx:
                out["only_anc_X"] += 1
                if any(lg):
                    out["only_anc_X_and_logical"] += 1
            elif any(lg):
                out["silent_logical"] += 1
                if len(examples) < 3:
                    examples.append((k, gadget[k], e, list(x[:5]), list(z[:5])))
            else:
                out["trivial"] += 1
    return out, examples


def ideal_cz(gadget):
    for xs in itertools.product(range(3), repeat=5):
        st = list(xs) + [0] * 6
        ph = 0
        for g, c, t in gadget:
            a, b = st[c], st[t]
            if g == "CSUM":
                st[t] = (a + b) % 3
            elif g == "CSUM_INV":
                st[t] = (b - a) % 3
            elif g == "CZ":
                ph += a * b
            elif g == "CZ_INV":
                ph -= a * b
        if st[:5] != list(xs) or any(st[5:]) or (ph - xs[2] * xs[3]) % 3:
            return False
    return True


# ---------------------------------------------------------------- C26
def single_errors():
    errs = []
    for q in range(5):
        for a, b in itertools.product(range(3), repeat=2):
            if a or b:
                x = [0] * 5
                z = [0] * 5
                x[q], z[q] = a, b
                errs.append((f"q{q}:X^{a}Z^{b}", (x, z)))
    return errs


def errors_on_23():
    errs = []
    for a, b, c, d in itertools.product(range(3), repeat=4):
        if a or b or c or d:
            x = [0, 0, a, c, 0]
            z = [0, 0, b, d, 0]
            errs.append((f"23:X2^{a}Z2^{b}X3^{c}Z3^{d}", (x, z)))
    return errs


def through_cz23(e):
    x, z = list(e[0]), list(e[1])
    v = SMAP["CZ"] @ np.array([x[2], z[2], x[3], z[3]]) % 3
    x[2], z[2], x[3], z[3] = [int(t) for t in v]
    return (x, z)


def ambiguity(errs):
    by = {}
    for name, e in errs:
        s = syndrome(e)
        if not any(s):
            continue
        by.setdefault(s, []).append((name, e))
    amb = {}
    for s, lst in by.items():
        ref = lst[0][1]
        classes = set()
        for name, e in lst:
            diff = (np.subtract(e[0], ref[0]), np.subtract(e[1], ref[1]))
            classes.add(logical(diff))
        amb[s] = (len(classes) > 1, [n for n, _ in lst])
    return amb


# ---------------------------------------------------------------- C4 (sdim)
def c4_sdim_check(n_seq=12, depth=6, shots=4, relabel=True, seed=7):
    from sdim.circuit import Circuit
    from sdim.program import Program
    from lrb import code_definitions as defns
    code = defns.FoldedQutritDetectionCode
    lib = defns.QutritCliffordLibrary
    inv = lib.build_inverse_lookup()
    cat = defns.single_qutrit_cliffords_exhaustive_strings
    rng = random.Random(seed)
    bad_checks = bad_logical = 0
    swapped_rounds = 0
    for _ in range(n_seq):
        cl = [rng.choice(cat) for _ in range(depth)]
        c = Circuit(dimension=3, num_qudits=9) + code.logical_plus_initial_state()
        frame = [0]

        def P(w):  # physical wire of a data/ancilla label in current frame
            if not frame[0]:
                return w
            return {2: 3, 3: 2, 5: 7, 7: 5, 6: 8, 8: 6}.get(w, w)

        def token(t):
            op = t[0]
            invt = len(t) > 1
            if op == "H":
                if invt:
                    c.add_gate("H", 1)
                    c.add_gate("H_INV", [0, 2, 3, 4])
                else:
                    c.add_gate("H", [0, 2, 3, 4])
                    c.add_gate("H_INV", 1)
                frame[0] ^= 1
            elif op == "P":
                if invt:
                    c.add_gate("CZ_INV", 2, 3)
                    c.add_gate("P", 1)
                    c.add_gate("P_INV", [0, 4])
                else:
                    c.add_gate("P", [0, 4])
                    c.add_gate("P_INV", 1)
                    c.add_gate("CZ", 2, 3)
            elif op in ("X", "Z"):
                wires = [0, 3] if op == "X" else [0, 2]
                c.add_gate(t, [P(w) if relabel else w for w in wires])

        def check_round():
            nonlocal swapped_rounds
            swapped_rounds += frame[0]
            R = (lambda w: P(w)) if relabel else (lambda w: w)
            for g, a, b in (("CNOT", 0, 5), ("CNOT_INV", 1, 5),
                            ("CNOT_INV", 3, 5)):
                c.add_gate(g, R(a), R(b))
            c.add_gate("M", R(5))
            for g, a, b in (("CNOT", 1, 6), ("CNOT", 2, 6),
                            ("CNOT_INV", 4, 6)):
                c.add_gate(g, R(a), R(b))
            c.add_gate("M", R(6))
            for anc, gs in ((7, (("CNOT", 7, 0), ("CNOT", 7, 1),
                                 ("CNOT_INV", 7, 2))),
                            (8, (("CNOT", 8, 3), ("CNOT_INV", 8, 1),
                                 ("CNOT_INV", 8, 4)))):
                c.add_gate("H", R(anc))
                for g, a, b in gs:
                    c.add_gate(g, R(a), R(b))
                c.add_gate("H_INV", R(anc))
                c.add_gate("M", R(anc))
            for a in (5, 6, 7, 8):
                c.add_gate("RESET", a)

        for cs in cl:
            for t in lib.qutrit_gate_seq_list(cs):
                token(t)
            check_round()
        for cs in reversed(cl):
            for t in inv[cs]:
                token(t)
        assert frame[0] == 0, "frame not restored"
        check_round()
        c.add_gate("H_INV", [0, 3])
        c.add_gate("M", [0, 3])
        for _s in range(shots):
            res = Program(c).simulate(shots=1, force_tableau=True)
            anc = [m for m in res if m.qudit_index >= 5]
            term = [m for m in res if m.qudit_index in (0, 3)][-2:]
            if any((not m.deterministic) or m.measurement_value % 3
                   for m in anc):
                bad_checks += 1
            if (term[0].measurement_value + term[1].measurement_value) % 3:
                bad_logical += 1
    return bad_checks, bad_logical, swapped_rounds


def main():
    print("== Conjugation rules derived from 9x9 matrices (v=(xc,zc,xt,zt)) ==")
    for g, M in SMAP.items():
        print(f"  {g}: M=\n{M}")

    print("\n== C6: CZ(2,3) gadgets ==")
    for name, gad in GADGETS.items():
        out, ex = classify(gad)
        print(f"  {name}: ideal CZ(2,3): {ideal_cz(gad)}; {out}")
        for e in ex:
            print("    silent example (after gate, fault, x_data, z_data):", e)

    print("\n== C26: syndrome ambiguity ==")
    s1 = single_errors()
    amb1 = ambiguity(s1)
    n_amb = sum(v[0] for v in amb1.values())
    print(f"  single-qutrit errors: {len(s1)}, distinct nonzero syndromes "
          f"{len(amb1)}, logically ambiguous {n_amb}")
    claimed_unamb = {s for s, v in amb1.items() if not v[0]}
    # circuit-level first-order set: singles + all errors on {2,3} + singles
    # on 2/3 pushed through CZ(2,3) (fault before a logical P).
    e23 = errors_on_23()
    pushed = [(n + "->CZ23", through_cz23(e)) for n, e in s1
              if n.startswith("q2") or n.startswith("q3")]
    amb2 = ambiguity(s1 + e23 + pushed)
    newly = [s for s in claimed_unamb if amb2[s][0]]
    print(f"  with errors on {{2,3}} and CZ(2,3)-propagated singles: of the "
          f"{len(claimed_unamb)} single-error syndromes called unambiguous, "
          f"{len(newly)} become ambiguous")
    for s in sorted(newly)[:8]:
        print("    syndrome", s, "shared by", amb2[s][1][:6])
    # two concrete witnesses
    x1 = ([0, 1, 0, 0, 0], [0] * 5)
    x23 = ([0, 0, 1, 1, 0], [0] * 5)
    print("  witness X1 vs X2X3: syndromes", syndrome(x1), syndrome(x23),
          "; logical class of quotient",
          logical((np.subtract(x23[0], x1[0]), np.subtract(x23[1], x1[1]))))
    x2 = ([0, 0, 1, 0, 0], [0] * 5)
    x2p = through_cz23(x2)
    x4z4 = ([0, 0, 0, 0, 2], [0, 0, 0, 0, 2])
    print("  witness X2 pushed through CZ(2,3) =", x2p, "syndrome",
          syndrome(x2p), "vs X4^-1 Z4^-1 syndrome", syndrome(x4z4),
          "; quotient logical",
          logical((np.subtract(x2p[0], x4z4[0]), np.subtract(x2p[1], x4z4[1]))))

    print("\n== C4: virtual SWAP in noiseless sdim LRB circuits ==")
    b1, b2, sw = c4_sdim_check(relabel=True)
    print(f"  relabelled checks/tokens: bad check shots {b1}, bad logical "
          f"shots {b2} (rounds run in swapped frame: {sw})")
    b1, b2, sw = c4_sdim_check(relabel=False)
    print(f"  negative control (no relabelling): bad check shots {b1}, bad "
          f"logical shots {b2}")

    print("\n== C17: relay starting in |s> (basis-state arithmetic) ==")
    ok_z = ok_x = True
    for s, xd, a0 in itertools.product(range(3), repeat=3):
        # Z-type relay: CSUM(d->r), CSUM(r->a), CSUM^-1(d->r)
        r = (s + xd) % 3
        a = (a0 + r) % 3
        r = (r - xd) % 3
        ok_z &= (r == s) and (a == (a0 + xd + s) % 3)
        # X-type fan-out relay, ancilla control value j (computational
        # branch): CSUM(a->r), CSUM(r->d), CSUM^-1(a->r)
        j = a0
        r = (s + j) % 3
        d = (xd + r) % 3
        r = (r - j) % 3
        ok_x &= (r == s) and (d == (xd + j + s) % 3)
    print("  Z relay: ancilla gets syndrome + s, relay returns to s:", ok_z)
    print("  X relay: data gets X^(j+s), relay returns to s:", ok_x)
    print("  (if the recorded s is wrong by e, the X relay leaves an "
          "unrecorded X^e on the data: a data error, not a frame shift)")


if __name__ == "__main__":
    main()
