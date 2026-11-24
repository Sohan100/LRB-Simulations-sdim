"""Exact leading-order logical error of a one-ancilla check round under the LRB
generator's SI1000 / SD6 noise, from fault pairs.

The circuit is the memory benchmark of ../../noise_bench.py: noiseless |+L> (and logical
H for the Z run), K noisy check rounds, noiseless (logical H^-1 and) X readout of the
data; a shot is kept if every ancilla readout is 0 and G3, G4 from the data readout are
0; the logical result is x0 + x3.

Noise: exactly LRBCircuitGenerator.apply_circuit_level_noise. moments() reproduces
schedule_moments (ASAP, collapse moments apart, readouts deferred to one final moment);
every moment adds DEP1 after 1-qutrit gates, DEP2 after CNOTs, an X^a flip after resets
and before readouts, and DEP1 on every untouched wire (idle rate in gate moments,
measure_reset_idle in reset / readout moments). DEP1 after a readout never reaches a
detector (the ancilla is reset first) and is left out.

Every fault is a Pauli at one noise location with probability rate * p / n (n = 8 for
DEP1, 80 for DEP2, 2 for a flip). Propagation is linear over F_3 (phases dropped), so a
fault's effect is a signature (all ancilla readout flips, final G3 and G4) and a shift
of the logical result. To leading order
    acceptance       = 1 - A1 p
    logical error    = L1 p + L2 p^2,
L1 summing faults with zero signature and a logical shift, L2 summing pairs of faults
at two different locations whose signatures cancel and whose logical shifts do not.
Pairs are counted exactly by grouping faults on (signature, shift).

Faults only pair within one round or across two adjacent rounds (a data error with a
syndrome flips every later readout), so L(K) is linear in K for K >= 2 and
L(3) - L(2) is the cost of one round inside a long sequence.

Usage: oa_pair_score.py [NAME ...]   (names of rounds.py ROUNDS, or "generator")
"""
import itertools
import sys

import numpy as np

import oa_symbolic as SY

D = 3
RATES = {
    "si1000": dict(g1=0.1, g2=1.0, reset=2.0, meas=5.0, idle=0.1, mr_idle=2.0),
    "sd6": dict(g1=1.0, g2=1.0, reset=1.0, meas=1.0, idle=1.0, mr_idle=1.0),
}
HBAR_INV = [("g", "H", 1), ("g", "H_INV", 0), ("g", "H_INV", 2), ("g", "H_INV", 3),
            ("g", "H_INV", 4), ("g", "SWAP", 2, 3)]
P1 = [(a, b) for a in range(D) for b in range(D) if (a, b) != (0, 0)]
COEF = {
    "flip": np.array([[1], [2]], dtype=np.int64),
    "dep1": np.array(P1, dtype=np.int64),
    "dep2": np.array([a + b for a in [(0, 0)] + P1 for b in [(0, 0)] + P1
                      if a + b != (0, 0, 0, 0)], dtype=np.int64),
}


def wires(op):
    return (op[1],) if op[0] in ("r", "m") else tuple(op[2:])


def moments(ops):
    """LRBCircuitGenerator.schedule_moments on a round op list: [(collapse, [ops])]."""
    last = {}
    for i, op in enumerate(ops):
        for w in wires(op):
            last[w] = i
    deferred = [i for i, op in enumerate(ops) if op[0] == "m" and last[op[1]] == i]
    out, ready = [], {}
    for i, op in enumerate(ops):
        if i in deferred:
            continue
        collapse = op[0] in ("r", "m")
        ws = wires(op)
        slot = max(ready.get(w, 0) for w in ws)
        while slot < len(out) and out[slot][0] != collapse:
            slot += 1
        if slot == len(out):
            out.append((collapse, []))
        out[slot][1].append(op)
        for w in ws:
            ready[w] = slot + 1
    if deferred:
        out.append((True, [ops[i] for i in deferred]))
    return out


def noisy_round(ops, nq=9):
    """Timeline of one round: ("op", op) and ("loc", kind, wires, rate key)."""
    tl = []
    for collapse, moment in moments(ops):
        busy = set()
        for op in moment:
            busy.update(wires(op))
            if op[0] == "m":
                tl.append(("loc", "flip", (op[1],), "meas"))
                tl.append(("op", op))
            elif op[0] == "r":
                tl.append(("op", op))
                tl.append(("loc", "flip", (op[1],), "reset"))
            elif len(op) == 3:
                tl.append(("op", op))
                tl.append(("loc", "dep1", (op[2],), "g1"))
            else:
                tl.append(("op", op))
                tl.append(("loc", "dep2", (op[2], op[3]), "g2"))
        for w in range(nq):
            if w not in busy:
                tl.append(("loc", "dep1", (w,), "mr_idle" if collapse else "idle"))
    return tl


def conj(op, x, z):
    if op[1] == "SWAP":
        a, b = op[2], op[3]
        x[a], x[b] = x[b], x[a]
        z[a], z[b] = z[b], z[a]
    else:
        SY.conj(op, x, z)


def basis_effects(ops, K, nq=9):
    """For every noise location of K rounds and every basis Pauli on its wires: the
    readout flips of all K rounds and the residual data error (x0..x4, z0..z4)."""
    tl = noisy_round(ops, nq) * K
    nmeas = sum(1 for it in tl if it[0] == "op" and it[1][0] == "m")
    locs = []
    for i, it in enumerate(tl):
        if it[0] != "loc":
            continue
        _, kind, ws, rate = it
        basis = []
        for w in ws:
            basis.append((w, 1, 0))
            if kind != "flip":
                basis.append((w, 0, 1))
        rows = []
        for w, bx, bz in basis:
            x, z = [0] * nq, [0] * nq
            x[w], z[w] = bx, bz
            flips = [0] * nmeas
            m = sum(1 for jt in tl[:i] if jt[0] == "op" and jt[1][0] == "m")
            for jt in tl[i + 1:]:
                if jt[0] != "op":
                    continue
                op = jt[1]
                if op[0] == "g":
                    conj(op, x, z)
                elif op[0] == "r":
                    x[op[1]] = z[op[1]] = 0
                else:
                    flips[m] = x[op[1]]
                    m += 1
            rows.append(flips + x[:5] + z[:5])
        locs.append((kind, rate, np.array(rows, dtype=np.int64)))
    return locs, nmeas


def terminal(run, res):
    """Final G3, G4 and logical shift from residual data errors res (n x 10: x, z)."""
    out = []
    for row in res:
        x, z = list(row[:5]) + [0] * 4, list(row[5:]) + [0] * 4
        if run == "Z":
            for op in HBAR_INV:
                conj(op, x, z)
        for q in range(5):
            conj(("g", "H_INV", q), x, z)
        f = x[:5]
        out.append(((f[0] + f[1] - f[2]) % D, (-f[1] + f[3] - f[4]) % D, (f[0] + f[3]) % D))
    return np.array(out, dtype=np.int64).reshape(-1, 3)


def score(ops, K, models=("si1000", "sd6")):
    """{(model, run): (A1, L1, L2)} for K noisy rounds."""
    locs, nmeas = basis_effects(ops, K)
    out = {}
    for run in ("X", "Z"):
        # per location: events (signature key, logical shift) and their multiplicities
        ev = []
        for kind, rate, B in locs:
            C = COEF[kind]
            E = (C @ B) % D
            term = terminal(run, E[:, nmeas:])
            sig = np.concatenate([E[:, :nmeas], term[:, :2]], axis=1)
            ev.append((kind, rate, sig, term[:, 2]))
        L = ev[0][2].shape[1]
        pw = D ** np.arange(L, dtype=np.int64)
        for model in models:
            R = RATES[model]
            keys, negk, logs, ws, lid = [], [], [], [], []
            for j, (kind, rate, sig, lg) in enumerate(ev):
                w = R[rate] / len(COEF[kind])
                keys.append(sig @ pw)
                negk.append(((-sig) % D) @ pw)
                logs.append(lg)
                ws.append(np.full(len(lg), w))
                lid.append(np.full(len(lg), j))
            keys, negk, logs, ws, lid = map(np.concatenate, (keys, negk, logs, ws, lid))
            A1 = ws[keys != 0].sum()
            L1 = ws[(keys == 0) & (logs != 0)].sum()
            L2 = (ordered_pairs(keys, negk, logs, ws) - same_loc(keys, negk, logs, ws, lid)) / 2
            out[(model, run)] = (A1, L1, L2)
    return out


def _table(keys, logs, ws):
    W = {}
    for k, l, w in zip(keys.tolist(), logs.tolist(), ws.tolist()):
        W.setdefault(k, [0.0, 0.0, 0.0])[l] += w
    return W


BAD = [(l1, l2) for l1 in range(D) for l2 in range(D) if (l1 + l2) % D]


def ordered_pairs(keys, negk, logs, ws):
    W = _table(keys, logs, ws)
    neg = dict(zip(keys.tolist(), negk.tolist()))
    tot = 0.0
    for k, row in W.items():
        other = W.get(neg[k])
        if other is None:
            continue
        tot += sum(row[a] * other[b] for a, b in BAD)
    return tot


def same_loc(keys, negk, logs, ws, lid):
    tot = 0.0
    bounds = np.flatnonzero(np.diff(lid)) + 1
    for sl in np.split(np.arange(len(lid)), bounds):
        tot += ordered_pairs(keys[sl], negk[sl], logs[sl], ws[sl])
    return tot


def per_round(ops, models=("si1000", "sd6")):
    """Cost of one round inside a long sequence: L(3) - L(2), summed over X and Z runs;
    A1 per round averaged over the two runs; L1 per round (0 for fault-safe rounds)."""
    s2, s3 = score(ops, 2, models), score(ops, 3, models)
    out = {}
    for m in models:
        A = sum(s3[(m, r)][0] - s2[(m, r)][0] for r in "XZ") / 2
        L1 = sum(s3[(m, r)][1] - s2[(m, r)][1] for r in "XZ")
        L2 = sum(s3[(m, r)][2] - s2[(m, r)][2] for r in "XZ")
        out[m] = (A, L1, L2)
    return out


def generator_ops():
    import oa_fault_spread as FSP
    return FSP.production_ops()


if __name__ == "__main__":
    import rounds as RR
    names = sys.argv[1:] or ["generator", "production-5", "production-5b"]
    for nm in names:
        ops = generator_ops() if nm == "generator" else RR.ROUNDS[nm]
        nmom = sum(1 for c, _ in moments(ops) if not c)
        for K in (1, 2, 3, 4):
            s = score(ops, K)
            cells = "  ".join(f"{m} {r}: A1 {s[(m, r)][0]:6.2f} L1 {s[(m, r)][1]:6.3f} L2 {s[(m, r)][2]:8.2f}"
                              for m in ("si1000", "sd6") for r in "XZ")
            print(f"{nm:14s} {nmom} gate moments K={K}  {cells}", flush=True)
        pr = per_round(ops)
        print(f"{nm:14s} per round: " + "  ".join(
            f"{m}: A1 {pr[m][0]:.2f} L1 {pr[m][1]:.3f} L2 {pr[m][2]:.2f}" for m in pr), flush=True)


def breakdown(ops, K, model, run):
    """L2 of K rounds split by the noise kinds (rate keys) of the two faults."""
    locs, nmeas = basis_effects(ops, K)
    R = RATES[model]
    pw = None
    per = {}
    for j, (kind, rate, B) in enumerate(locs):
        E = (COEF[kind] @ B) % D
        term = terminal(run, E[:, nmeas:])
        sig = np.concatenate([E[:, :nmeas], term[:, :2]], axis=1)
        if pw is None:
            pw = D ** np.arange(sig.shape[1], dtype=np.int64)
        w = R[rate] / len(COEF[kind])
        d = per.setdefault(rate, [[], [], [], [], []])
        d[0].append(sig @ pw); d[1].append(((-sig) % D) @ pw); d[2].append(term[:, 2])
        d[3].append(np.full(len(E), w)); d[4].append(np.full(len(E), j))
    per = {k: [np.concatenate(v) for v in d] for k, d in per.items()}
    out = {}
    kinds = sorted(per)
    for i, a in enumerate(kinds):
        for b in kinds[i:]:
            A, B_ = per[a], per[b]
            if a == b:
                val = (ordered_pairs(A[0], A[1], A[2], A[3]) - same_loc(*A[:4], A[4])) / 2
            else:
                keys = np.concatenate([A[0], B_[0]]); neg = np.concatenate([A[1], B_[1]])
                lg = np.concatenate([A[2], B_[2]]); ws = np.concatenate([A[3], B_[3]])
                tot = ordered_pairs(keys, neg, lg, ws)
                val = (tot - ordered_pairs(*A[:4]) - ordered_pairs(*B_[:4])) / 2
            out[(a, b)] = val
    return out
