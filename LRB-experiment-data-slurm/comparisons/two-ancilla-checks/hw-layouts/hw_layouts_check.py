"""Check rounds for the two hardware layouts (user's figure, 2026-11-16).

Stabilizers (code labels, unchanged):
  G1 = Z0 Z1^-1 Z3^-1   G2 = Z1 Z2 Z4^-1   G3 = X0 X1 X2^-1   G4 = X1^-1 X3 X4^-1

Left layout:  a -- q0 q1 q2 q3,  b -- q2 q4,  data coupler q2 -- q3.
Right layout: a -- q0 q1 q2,  b -- q0 q3,  c -- q1 q2 q4,  data coupler q2 -- q3.

Only gates on drawn couplers. One stabilizer per step. A split check uses a
data relay: the lender ancilla collects its part, lends it to a data qutrit the
reader also touches, the reader reads, the lender takes it back and clears.

Tests:
  1. zero noise: every readout 0 for 5 rounds, logical kept (also with logical H);
  2. every single-qutrit data Pauli before round 1: each readout deterministic,
     identical in rounds 1 and 2 (a QND stabilizer readout, not a partial
     measurement), equal to +-(the syndrome it should carry), lender reads 0;
  3. exhaustive single faults (as fault_check.py): silent logical errors.
"""
import itertools
import sys
from collections import Counter
from multiprocessing import Pool

from sdim.program import Program
from lrb.code_definitions import FoldedQutritDetectionCode as CODE

PAULIS = [(x, z) for x in range(3) for z in range(3) if (x, z) != (0, 0)]
A, B, C = 5, 6, 7


def g(name, *w): return ("g", name, *w)
def m(w, tag): return ("m", w, tag)          # tag: check name, or "relay"
def r(*ws): return [("r", w) for w in ws]


# ---------- stabilizer syndromes of a data Pauli (bilinear, exponents) ----------
STAB = {  # (type, exponent vector over q0..q4)
    "G1": ("Z", (1, -1, 0, -1, 0)), "G2": ("Z", (0, 1, 1, 0, -1)),
    "G3": ("X", (1, 1, -1, 0, 0)), "G4": ("X", (0, -1, 0, 1, -1)),
    "G3G4": ("X", (1, 0, -1, 1, -1)),
    "G3G4inv": ("X", (1, -1, -1, -1, 1)),     # G3 G4^-1
    "G3invG4": ("X", (-1, 1, 1, 1, -1)),      # G3^-1 G4
}


def syndrome(check, wire, x, z):
    """Commutation phase exponent of X^x Z^z on `wire` with the check (mod 3)."""
    typ, vec = STAB[check]
    return (vec[wire] * (x if typ == "Z" else -z)) % 3


# ---------- building blocks ----------
def z_direct(anc, coeffs, tag):
    return [g("CNOT" if k == 1 else "CNOT_INV", q, anc) for q, k in coeffs] + [m(anc, tag), *r(anc)]


def x_direct(anc, coeffs, tag):
    return ([g("H", anc)] + [g("CNOT" if k == 1 else "CNOT_INV", anc, q) for q, k in coeffs]
            + [g("H_INV", anc), m(anc, tag), *r(anc)])


def cn(k): return "CNOT" if k % 3 == 1 else "CNOT_INV"


def z_relay(reader, reads, lender, src, p, tag):
    """reads: [(q, coeff)] incl. the shared qutrit p; src: (q, coeff) of lender part."""
    cp = dict(reads)[p]
    s, cs = src
    k = cs * cp
    ops = [g(cn(k), s, lender), g("CNOT", lender, p)]
    ops += [g(cn(c), q, reader) for q, c in reads]
    ops += [g("CNOT_INV", lender, p), g(cn(-k), s, lender)]
    return ops + [m(reader, tag), m(lender, "relay"), *r(reader, lender)]


def x_relay_lend(lender, src, p, cp):
    """src: (q, coeff) or a list of them. The hook of this lend is X_p prod X_s^(c_s c_p)."""
    srcs = [src] if isinstance(src[0], int) else list(src)
    pre = [g("H", lender)] + [g(cn(cs * cp), lender, s) for s, cs in srcs] + [g("CNOT", p, lender)]
    post = ([g("CNOT_INV", p, lender)] + [g(cn(-cs * cp), lender, s) for s, cs in reversed(srcs)]
            + [g("H_INV", lender)])
    return pre, post


def x_relay(reader, reads, lenders, tag):
    """lenders: [(lender, (src, coeff), p)]; p must be in reads."""
    rd = dict(reads)
    pres, posts = [], []
    for L, src, p in lenders:
        pre, post = x_relay_lend(L, src, p, rd[p])
        pres += pre
        posts += post
    ops = pres + [g("H", reader)] + [g(cn(c), reader, q) for q, c in reads] + [g("H_INV", reader)]
    ops += posts
    return ops + [m(reader, tag)] + [m(L, "relay") for L, _, _ in lenders] + r(reader, *[L for L, _, _ in lenders])


# ---------- left layout ----------
L_G1 = z_direct(A, [(0, 1), (1, -1), (3, -1)], "G1")
L_G3 = x_direct(A, [(0, 1), (1, 1), (2, -1)], "G3")
L_G2_readA = z_relay(A, [(1, 1), (2, 1)], B, (4, -1), 2, "G2")      # b lends -z4 via q2
L_G2_readB = z_relay(B, [(2, 1), (4, -1)], A, (1, 1), 2, "G2")      # a lends z1 via q2
L_G34 = x_relay(A, [(0, 1), (2, -1), (3, 1)], [(B, (4, -1), 2)], "G3G4")  # b lends x4 via q2
# hook-safe X relays (hook X_p prod X_s^(c_s c_p) must be seen by a Z check)
L_G34_readB = x_relay(B, [(2, -1), (4, -1)], [(A, [(0, 1), (3, 1)], 2)], "G3G4")   # hook X2 X0^-1 X3^-1
L_G34inv_readA = x_relay(A, [(0, 1), (1, -1), (2, -1), (3, -1)], [(B, (4, 1), 2)], "G3G4inv")  # hook X2 X4^-1
L_G3invG4_readA = x_relay(A, [(0, -1), (1, 1), (2, 1), (3, 1)], [(B, (4, -1), 2)], "G3invG4")  # hook X2 X4^-1
# G4 itself: q2 is not in G4, so a reads q2 before and after the lend (difference)
L_G4_diff = [g("H", B), g("CNOT_INV", B, 4), g("H", A), g("CNOT_INV", A, 1), g("CNOT", A, 3),
             g("CNOT_INV", A, 2), g("CNOT", 2, B), g("CNOT", A, 2), g("CNOT_INV", 2, B),
             g("CNOT", B, 4), g("H_INV", A), g("H_INV", B), m(A, "G4"), m(B, "relay"), *r(A, B)]

# hybrid (uses the q2-q3 data coupler): b reads q4 and q2; q2 carries a's part
# (lent by a) and q3's part (moved by a data CNOT), so no gate holds both halves
# of X2 X4.
L_G4_hyb = [g("H", B), g("CNOT_INV", B, 4), g("CNOT", B, 2), g("H", A), g("CNOT", A, 1),
            g("CNOT", 2, A), g("CNOT", 2, 3), g("CNOT_INV", B, 2), g("CNOT_INV", 2, 3),
            g("CNOT_INV", 2, A), g("CNOT_INV", A, 1), g("H_INV", A), g("H_INV", B),
            m(B, "G4"), m(A, "relay"), *r(A, B)]
L_G34inv_hyb = [g("H", B), g("CNOT", B, 4), g("H", A), g("CNOT_INV", A, 0), g("CNOT", A, 1),
                g("CNOT", 2, A), g("CNOT_INV", 2, 3), g("CNOT_INV", B, 2), g("CNOT", 2, 3),
                g("CNOT_INV", 2, A), g("CNOT_INV", A, 1), g("CNOT", A, 0), g("H_INV", A),
                g("H_INV", B), m(B, "G3G4inv"), m(A, "relay"), *r(A, B)]

# ---------- right layout ----------
R_G3 = x_direct(A, [(0, 1), (1, 1), (2, -1)], "G3")
R_G2 = z_direct(C, [(1, 1), (2, 1), (4, -1)], "G2")
R_G1_readA = z_relay(A, [(0, 1), (1, -1)], B, (3, -1), 0, "G1")     # b lends -z3 via q0
R_G1_readB = z_relay(B, [(0, 1), (3, -1)], A, (1, -1), 0, "G1")     # a lends -z1 via q0
# G4 family needs q3 (only b) and q4 (only c); b and c share no data qutrit.
# (i) three ancillas: a reads G3G4 = X0 X2^-1 X3 X4^-1, b lends x3 via q0, c lends x4 via q2
R_G34_3anc = x_relay(A, [(0, 1), (2, -1)], [(B, (3, 1), 0), (C, (4, -1), 2)], "G3G4")
# hook-safe: a reads G3 G4^-1 (or G3^-1 G4); hooks X0 X3^-1 and X2 X4^-1 (or X1 X4^..)
R_G34inv_3anc = x_relay(A, [(0, 1), (1, -1), (2, -1)], [(B, (3, -1), 0), (C, (4, 1), 2)], "G3G4inv")
R_G34inv_3anc_p1 = x_relay(A, [(0, 1), (1, -1), (2, -1)], [(B, (3, -1), 0), (C, (4, 1), 1)], "G3G4inv")
R_G3invG4_3anc = x_relay(A, [(0, -1), (1, 1), (2, 1)], [(B, (3, 1), 0), (C, (4, -1), 2)], "G3invG4")
# (ii) data coupler q2-q3: c reads X1^-1 X4^-1 and q2 before/after CNOT 2->3 (difference)
R_G4_dd = [g("H", C), g("CNOT_INV", C, 1), g("CNOT_INV", C, 4), g("CNOT", C, 2), g("CNOT", 2, 3),
           g("CNOT_INV", C, 2), g("CNOT_INV", 2, 3), g("H_INV", C), m(C, "G4"), *r(C)]

ONE_ANCILLA = (z_direct(5, [(0, 1), (1, -1), (3, -1)], "G1") + z_direct(6, [(1, 1), (2, 1), (4, -1)], "G2")
               + x_direct(7, [(0, 1), (1, 1), (2, -1)], "G3") + x_direct(8, [(3, 1), (1, -1), (4, -1)], "G4"))

SCHEMES = {
    "one ancilla (reference)": ONE_ANCILLA,
    "LEFT  G1,G2(a reads),G3,G3G4": L_G1 + L_G2_readA + L_G3 + L_G34,
    "LEFT  G1,G2(b reads),G3,G3G4": L_G1 + L_G2_readB + L_G3 + L_G34,
    "LEFT  G1,G2(a reads),G3,G4-diff": L_G1 + L_G2_readA + L_G3 + L_G4_diff,
    "RIGHT G1(a reads),G2,G3,G3G4 3-anc": R_G1_readA + R_G2 + R_G3 + R_G34_3anc,
    "RIGHT G1(b reads),G2,G3,G3G4 3-anc": R_G1_readB + R_G2 + R_G3 + R_G34_3anc,
    "RIGHT G1(a reads),G2,G3,G4 via q2-q3": R_G1_readA + R_G2 + R_G3 + R_G4_dd,
    "LEFT  G1,G2(a reads),G3,G3G4(b reads, a lends)": L_G1 + L_G2_readA + L_G3 + L_G34_readB,
    "LEFT  G1,G2(b reads),G3,G3G4(b reads, a lends)": L_G1 + L_G2_readB + L_G3 + L_G34_readB,
    "LEFT  G1,G2(a reads),G3,G3G4^-1(a reads)": L_G1 + L_G2_readA + L_G3 + L_G34inv_readA,
    "LEFT  G1,G2(a reads),G3,G3^-1G4(a reads)": L_G1 + L_G2_readA + L_G3 + L_G3invG4_readA,
    "RIGHT G1(a reads),G2,G3,G3G4^-1 3-anc": R_G1_readA + R_G2 + R_G3 + R_G34inv_3anc,
    "RIGHT G1(a reads),G2,G3,G3G4^-1 3-anc c via q1": R_G1_readA + R_G2 + R_G3 + R_G34inv_3anc_p1,
    "RIGHT G1(a reads),G2,G3,G3^-1G4 3-anc": R_G1_readA + R_G2 + R_G3 + R_G3invG4_3anc,
    "LEFT  G1,G2(a reads),G3,G4 hybrid q2-q3": L_G1 + L_G2_readA + L_G3 + L_G4_hyb,
    "LEFT  G1,G2(a reads),G3,G3G4^-1 hybrid q2-q3": L_G1 + L_G2_readA + L_G3 + L_G34inv_hyb,
    "LEFT  G1,G2(b reads),G3,G4 hybrid q2-q3": L_G1 + L_G2_readB + L_G3 + L_G4_hyb,
}


# ---------- simulation ----------
def build(ops, rounds, fault=None, pre_error=None, sandwich=False):
    c = CODE.logical_plus_initial_state()
    if sandwich:
        c = c + CODE.single_qudit_logical_gate_circuit("H")
    labels, count = [], Counter()

    def inject(paulis):
        for w, (x, z) in paulis.items():
            for _ in range(x):
                c.add_gate("X", w)
            for _ in range(z):
                c.add_gate("Z", w)

    if pre_error:
        inject(pre_error)
    for rnd in range(rounds):
        if rnd == 0 and fault and fault[0] == -1:
            inject(fault[1])
        for i, op in enumerate(ops):
            if op[0] == "g":
                c.add_gate(op[1], *op[2:])
            elif op[0] == "m":
                c.add_gate("M", op[1])
                labels.append((rnd, op[2], op[1], count[op[1]], i))
                count[op[1]] += 1
            else:
                c.add_gate("RESET", op[1])
                count[op[1]] += 1
            if rnd == 0 and fault and fault[0] == i:
                inject(fault[1])
    if sandwich:
        c = c + CODE.single_qudit_logical_gate_circuit("H_INV")
    for q in range(5):
        c.add_gate("H_INV", q)
        c.add_gate("M", q)
        labels.append(("final", "data", q, count[q], -1))
        count[q] += 1
    return c, labels


def run(c, labels, shots):
    res = Program(c).simulate(shots=max(shots, 2))[0]
    return {lab: [int(res[lab[2]][lab[3]][s].measurement_value) for s in range(shots)] for lab in labels}


def two_qudit_and_sites(ops):
    sites = [(-1, (q,)) for q in range(CODE.num_qudits)]
    for i, op in enumerate(ops):
        if op[0] == "g":
            sites.append((i, tuple(op[2:])))
        elif op[0] == "r":
            sites.append((i, (op[1],)))
    return sites


def faults(ops):
    for i, wires in two_qudit_and_sites(ops):
        for combo in itertools.product([(0, 0)] + PAULIS, repeat=len(wires)):
            if any(p != (0, 0) for p in combo):
                yield i, dict(zip(wires, combo))


def classify(vals, s=0):
    final = {lab[2]: v[s] for lab, v in vals.items() if lab[0] == "final"}
    x = [final[q] for q in range(5)]
    anc = {lab: v[s] for lab, v in vals.items() if lab[0] != "final"}
    chk = any(v for lab, v in anc.items() if lab[1] != "relay")
    rel = any(v for lab, v in anc.items() if lab[1] == "relay")
    dat = bool((x[0] + x[1] - x[2]) % 3 or (-x[1] + x[3] - x[4]) % 3)
    return chk, rel, dat, (x[0] + x[3]) % 3


def stats(ops):
    n2 = sum(1 for op in ops if op[0] == "g" and len(op) == 4)
    nm = sum(1 for op in ops if op[0] == "m")
    return n2, nm


def noiseless_tests(name, ops):
    out = []
    for sandwich in (False, True):
        vals = run(*build(ops, 5, sandwich=sandwich), shots=50)
        bad = [lab for lab, v in vals.items() if lab[0] != "final" and any(v)]
        lg = [classify(vals, s) for s in range(50)]
        ok = not bad and all(t == (False, False, False, 0) for t in lg)
        out.append(f"  zero noise, 5 rounds{' (H sandwich)' if sandwich else ''}: {'OK' if ok else 'FAIL ' + str(bad[:3])}")
    signs, problems = {}, []
    for wire in range(5):
        for (x, z) in PAULIS:
            vals = run(*build(ops, 2, pre_error={wire: (x, z)}), shots=30)
            for lab, v in vals.items():
                if lab[0] == "final":
                    continue
                if len(set(v)) != 1:
                    problems.append(f"random {lab[1]} r{lab[0]} for X^{x}Z^{z} on q{wire}")
                    continue
                val = v[0]
                if lab[1] == "relay":
                    if val:
                        problems.append(f"relay reads {val} for X^{x}Z^{z} on q{wire}")
                    continue
                exp = syndrome(lab[1], wire, x, z)
                key = (lab[1], lab[4])
                if exp == 0:
                    if val:
                        problems.append(f"{lab[1]} reads {val}, expected 0, X^{x}Z^{z} q{wire}")
                    continue
                sgn = 1 if val == exp else (-1 if val == (-exp) % 3 else None)
                if sgn is None:
                    problems.append(f"{lab[1]} reads {val} vs {exp}")
                elif signs.setdefault(key, sgn) != sgn:
                    problems.append(f"{lab[1]} sign flips")
            # QND: rounds 1 and 2 agree
            r0 = {(lab[1], lab[4]): v[0] for lab, v in vals.items() if lab[0] == 0}
            r1 = {(lab[1], lab[4]): v[0] for lab, v in vals.items() if lab[0] == 1}
            if r0 != r1:
                problems.append(f"round 1 != round 2 for X^{x}Z^{z} on q{wire}")
    sg = ", ".join(f"{k[0]}:{'+' if s == 1 else '-'}" for k, s in sorted(signs.items(), key=lambda t: t[0][1]))
    out.append(f"  40 single data Paulis: {'OK' if not problems else 'FAIL'}; readout signs {sg}")
    out += ["    " + p for p in problems[:6]]
    return out


def fault_test(args):
    name, ops = args
    lines = []
    for sandwich in (False, True):
        total = silent = silent_wo = 0
        ex = []
        for f in faults(ops):
            total += 1
            vals = run(*build(ops, 2, fault=f, sandwich=sandwich), shots=2)
            chk, rel, dat, lg = classify(vals)
            if lg and not chk and not dat:
                silent_wo += 1
                if not rel:
                    silent += 1
                    ex.append(f)
        lines.append(f"  single faults{' (H sandwich)' if sandwich else ''}: {total} faults, "
                     f"{silent} silent logical ({silent_wo} if relay readouts ignored)"
                     + (f"; e.g. {ex[:3]}" if ex else ""))
    return name, lines


def naive_right_g4():
    """Negative control: b reads x3, c reads -x1 -x4, add classically (no relay)."""
    half = [g("H", B), g("CNOT", B, 3), g("H_INV", B), g("H", C), g("CNOT_INV", C, 1),
            g("CNOT_INV", C, 4), g("H_INV", C), m(B, "hb"), m(C, "hc"), *r(B, C)]
    ops = R_G1_readA + R_G2 + R_G3 + half
    shots = 3000
    c, labels = build(ops, 5)
    vals = run(c, labels, shots)
    by = {(lab[0], lab[1]): v for lab, v in vals.items() if lab[0] != "final"}
    allzero = [all(by[(rd, t)][s] == 0 for t in ("G1", "G2", "G3") for rd in range(k))
               and all((by[(rd, "hb")][s] + by[(rd, "hc")][s]) % 3 == 0 for rd in range(k))
               for k in (1, 2, 5) for s in [None] if True for s in range(shots)]
    n = shots
    pass_k = [sum(allzero[i * n:(i + 1) * n]) for i in range(3)]
    return (f"RIGHT naive split of G4 over b and c (classical sum): all checks 0 in "
            f"{pass_k[0]}/{n} shots after 1 round, {pass_k[1]}/{n} after 2, {pass_k[2]}/{n} after 5")


if __name__ == "__main__":
    only = [a.split("=", 1)[1] for a in sys.argv if a.startswith("--only=")]
    SELECTED = {k: v for k, v in SCHEMES.items() if not only or any(o in k for o in only)}
    if not only:
        print(naive_right_g4(), flush=True)
    for name, ops in SELECTED.items():
        n2, nm = stats(ops)
        print(f"{name}: {n2} two-qudit gates, {nm} measurements per round", flush=True)
        for line in noiseless_tests(name, ops):
            print(line, flush=True)
    if "--faults" in sys.argv:
        with Pool(len(SCHEMES)) as pool:
            for name, lines in pool.imap(fault_test, SELECTED.items()):
                print(name, flush=True)
                for line in lines:
                    print(line, flush=True)
