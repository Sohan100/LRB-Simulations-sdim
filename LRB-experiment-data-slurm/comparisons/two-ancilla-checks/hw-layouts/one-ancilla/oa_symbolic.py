"""Fast symplectic (tableau) screening of check rounds on 9 qutrits (F_3, phases dropped).

Used only to screen many op lists quickly; every round reported in the README is
re-checked with the sdim-based tests of ../hw_layouts_check.py.

- round_ok(ops): for each measured ancilla a, Z_a pulled back to the start of the
  round has a data part equal to a nonzero multiple of its check, no X part on any
  ancilla, and every data Pauli passes through the round with its data part unchanged
  (non-destructive, QND).
- silent(ops): exhaustive single faults (every non-identity Pauli on the wires of
  every gate and reset, after it, and on every data qutrit at round start), as in
  hw_layouts_check.faults; a fault is silent if it flips no readout of this round or
  of an ideal next round and leaves a nontrivial logical on the data.
"""
import itertools

D = 3
NQ = 9
CHK = {"G1": ("Z", (1, 2, 0, 2, 0)), "G2": ("Z", (0, 1, 1, 0, 2)),
       "G3": ("X", (1, 1, 2, 0, 0)), "G4": ("X", (0, 2, 0, 1, 2))}


def stab_vecs():
    out = []
    for nm, (k, v) in CHK.items():
        x = list(v) if k == "X" else [0] * 5
        z = list(v) if k == "Z" else [0] * 5
        out.append((tuple(x), tuple(z)))
    return out


STAB = stab_vecs()


def sympl(a, b):
    """Commutation exponent of data Paulis a=(x,z), b=(x,z)."""
    return (sum(a[0][i] * b[1][i] - a[1][i] * b[0][i] for i in range(5))) % D


def span(vecs):
    out = set()
    for cs in itertools.product(range(D), repeat=len(vecs)):
        x = tuple(sum(c * v[0][i] for c, v in zip(cs, vecs)) % D for i in range(5))
        z = tuple(sum(c * v[1][i] for c, v in zip(cs, vecs)) % D for i in range(5))
        out.add((x, z))
    return out


STAB_GROUP = span(STAB)


def conj(gate, x, z, inverse=False):
    """Forward conjugation g P g^dag of P = X^x Z^z (lists, modified in place)."""
    name = gate[1]
    if name in ("CNOT", "CNOT_INV"):
        s = 1 if name == "CNOT" else 2
        if inverse:
            s = (-s) % D
        c, t = gate[2], gate[3]
        x[t] = (x[t] + s * x[c]) % D
        z[c] = (z[c] - s * z[t]) % D
    elif name in ("H", "H_INV"):
        h = name == "H"
        if inverse:
            h = not h
        q = gate[2]
        if h:      # X -> Z, Z -> X^-1
            x[q], z[q] = (-z[q]) % D, x[q]
        else:      # X -> Z^-1, Z -> X
            x[q], z[q] = z[q], (-x[q]) % D
    else:
        raise ValueError(name)


def gates_only(ops):
    return [op for op in ops if op[0] == "g"]


def round_ok(ops):
    gs = gates_only(ops)
    meas = [(op[1], op[2]) for op in ops if op[0] == "m"]
    for a, tag in meas:
        x, z = [0] * NQ, [0] * NQ
        z[a] = 1
        for g in reversed(gs):
            conj(g, x, z, inverse=True)
        if any(x[w] for w in range(5, NQ)):
            return False
        kind, v = CHK[tag]
        dx, dz = x[:5], z[:5]
        ok = False
        for c in (1, 2):
            want = [(c * e) % D for e in v]
            if kind == "Z" and dz == want and not any(dx):
                ok = True
            if kind == "X" and dx == want and not any(dz):
                ok = True
        if not ok:
            return False
    for q in range(5):
        for xx, zz in ((1, 0), (0, 1)):
            x, z = [0] * NQ, [0] * NQ
            x[q], z[q] = xx, zz
            for g in gs:
                conj(g, x, z)
            want_x = [int(i == q) * xx for i in range(5)]
            want_z = [int(i == q) * zz for i in range(5)]
            if x[:5] != want_x or z[:5] != want_z:
                return False
    return True


PAULIS = [(a, b) for a in range(D) for b in range(D) if (a, b) != (0, 0)]


def fault_sites(ops):
    sites = [(-1, (q,)) for q in range(5)]
    for i, op in enumerate(ops):
        if op[0] == "g":
            sites.append((i, tuple(op[2:])))
        elif op[0] == "r":
            sites.append((i, (op[1],)))
    return sites


def silent(ops, examples=False):
    meas = {i: op[1] for i, op in enumerate(ops) if op[0] == "m"}
    total = bad = 0
    ex = []
    for i, wires in fault_sites(ops):
        for combo in itertools.product([(0, 0)] + PAULIS, repeat=len(wires)):
            if all(p == (0, 0) for p in combo):
                continue
            total += 1
            x, z = [0] * NQ, [0] * NQ
            for w, (a, b) in zip(wires, combo):
                x[w], z[w] = a, b
            flipped = False
            for j in range(i + 1, len(ops)):
                op = ops[j]
                if op[0] == "g":
                    conj(op, x, z)
                elif op[0] == "r":
                    x[op[1]] = z[op[1]] = 0
                elif op[0] == "m":
                    if x[op[1]]:
                        flipped = True
            if flipped:
                continue
            E = (tuple(x[:5]), tuple(z[:5]))
            if any(sympl(E, s) for s in STAB):
                continue          # the next ideal round (or terminal check) sees it
            if E in STAB_GROUP:
                continue
            bad += 1
            if examples and len(ex) < 5:
                ex.append((i, dict(zip(wires, combo))))
    return (bad, total, ex) if examples else (bad, total)
