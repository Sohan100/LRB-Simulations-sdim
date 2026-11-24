"""Pick and verify the best 8-gate encoder for the paper's one-ancilla layout.

All minimal CSS encoders through ancillas a1 (wire 5: q0,q1,q3) and a2 (wire 6:
q1,q2,q4) from ancilla_encoder_search; for each: circuit depth (H layer + CNOT
layers, ASAP), validity for the qubit code (same gates, CNOT_INV -> CNOT, mod 2),
and an exact 9-qudit statevector check that the data end in the paper encoder's
|+_L> and both ancillas in |0>.
"""
import sys

import numpy as np

sys.path.insert(0, ".")
import ancilla_encoder_search as AS  # noqa: E402
import tree_encoders as TE  # noqa: E402

L = AS.LAYOUTS["one-ancilla"]
SUB = (5, 6)


def depth(S, gates):
    busy = {q: 1 for q in S}
    for _, c, t in gates:
        layer = max(busy.get(c, 0), busy.get(t, 0)) + 1
        busy[c] = busy[t] = layer
    return max(busy.values())


def maps_ok(S, gates, d, plus):
    n = 9
    M = np.zeros((len(S), n), dtype=int)
    for i, q in enumerate(S):
        M[i, q] = 1
    for name, c, t in gates:
        s = 1 if (name == "CNOT" or d == 2) else -1
        M[:, t] = (M[:, t] + s * M[:, c]) % d
    V = np.array([AS.ET.G3, AS.ET.G4, AS.ET.XL]) % d
    tgt = [list(v) + [0] * 4 for v in V.tolist()] + [[int(i == a) for i in range(n)] for a in plus]
    return TE.rref_mod(M, d) == TE.rref_mod(np.array(tgt), d)


def statevector_check(S, gates, d):
    nq = 9
    w = np.exp(2j * np.pi / d)
    F = np.array([[w ** (j * k) for j in range(d)] for k in range(d)]) / np.sqrt(d)
    psi = np.zeros((d,) * nq, dtype=complex)
    psi[(0,) * nq] = 1
    def h(psi, q):
        return np.moveaxis(np.tensordot(F, psi, axes=([1], [q])), 0, q)
    def cx(psi, c, t, s):
        out = np.empty_like(psi)
        for v in range(d):
            idx = [slice(None)] * nq
            idx[c] = v
            out[tuple(idx)] = np.roll(psi[tuple(idx)], s * v, axis=t if t < c else t - 1)
        return out
    for q in S:
        psi = h(psi, q)
    for name, c, t in gates:
        psi = cx(psi, c, t, 1 if name == "CNOT" else -1)
    ref = np.zeros((d,) * nq, dtype=complex)
    ref[(0,) * nq] = 1
    # paper encoder: H_INV on 2,3,4; CNOT_INV 2->1; CNOT 1->0; CNOT 3->0; CNOT 4->2 (H_INV|0> = H|0>)
    for q in (2, 3, 4):
        ref = h(ref, q)
    for name, c, t in [("CNOT_INV", 2, 1), ("CNOT", 1, 0), ("CNOT", 3, 0), ("CNOT", 4, 2)]:
        ref = cx(ref, c, t, 1 if name == "CNOT" else -1)
    return abs(np.vdot(ref, psi)) ** 2


if __name__ == "__main__":
    couplers = [e for e in L["couplers"] if all(w < 5 or w in SUB for w in e)]
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        dep, hits = AS.search({"n": 9, "anc": SUB, "couplers": couplers}, 8)
    rows = []
    for W, plus, path in hits:
        S = tuple(sorted(next(i for i, v in enumerate(row) if v) for row in W))
        rows.append((depth(S, path), S, plus, path, maps_ok(S, path, 2, plus)))
    rows.sort(key=lambda r: (not r[4], r[0]))
    print(f"{len(rows)} eight-gate encoders through a1, a2; {sum(r[4] for r in rows)} also valid for qubits; "
          f"depths {sorted(set(r[0] for r in rows))}")
    for dpt, S, plus, path, q2 in rows[:6]:
        f3 = statevector_check(S, path, 3)
        f2 = statevector_check(S, path, 2) if q2 else float("nan")
        print(f"  depth {dpt}  |+> on {S}  ancillas end in |0>  gates {list(path)}  "
              f"overlap with paper |+_L>|0,0>: d=3 {f3:.12f}, d=2 {f2:.12f}")
