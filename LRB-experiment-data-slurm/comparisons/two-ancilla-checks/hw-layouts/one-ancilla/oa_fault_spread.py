"""How far do faults spread, and which fault pairs give an undetected logical error?

Tableau propagation (oa_symbolic.conj, F_3, phases dropped), same fault locations as
the sdim checker: every non-identity Pauli on the wires of every gate (after it) and of
every reset, and on every data qutrit at round start. For each single fault:
- detected: it flips a readout of this round or leaves a data error with a nonzero
  syndrome (the next round or the terminal check sees it);
- spread: number of data qutrits the residual data error touches.
Propagation is linear, so a pair of faults at two different locations is silent iff the
two readout-flip vectors cancel, the two syndromes cancel, and the summed residual is a
nontrivial logical. Pairs are counted exactly with a hash on (flips, syndrome, logical
class), split by the location types of the two faults. With two-qutrit depolarizing
noise of strength p (p/80 per non-identity two-qutrit Pauli) the leading logical error
of a round from two-qutrit-gate faults is about N_pairs(2q, 2q) (p/80)^2.
Usage: oa_fault_spread.py
"""
import itertools
from collections import Counter, defaultdict

import oa_symbolic as SY
import rounds as R


def production_ops():
    """The LRB generator's one-ancilla round as an op list (generator order)."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
    import noise_bench as NB
    import protocol_search as PS
    circ, _ = NB.production_round("folded_qutrit", "si1000")
    tags = {5: "G1", 6: "G2", 7: "G3", 8: "G4"}
    ops = []
    for o in circ.operations:
        w = tuple(PS.GEN._operation_qudits(o))
        if o.gate_name == "RESET":
            ops.append(("r", w[0]))
        elif o.gate_name == "M":
            ops.append(("m", w[0], tags[w[0]]))
        else:
            ops.append(("g", o.gate_name, *w))
    return ops

D = 3
PAULIS = [(a, b) for a in range(D) for b in range(D) if (a, b) != (0, 0)]
XBAR = ((1, 0, 0, 1, 0), (0,) * 5)          # X0 X3 (rows: x, z)


def zbar():
    """A Z-type logical: commutes with G3, G4, does not commute with X0 X3."""
    for z in itertools.product(range(D), repeat=5):
        E = ((0,) * 5, z)
        if all(SY.sympl(E, s) == 0 for s in SY.STAB) and SY.sympl(E, XBAR):
            return E


ZBAR = zbar()


def kind(op):
    if op is None:
        return "data at start"
    if op[0] == "r":
        return "reset"
    if len(op) == 3:
        return "1q gate"
    c, t = op[2], op[3]
    if c < 5 and t < 5:
        return "data-data CNOT"
    if c >= 5 and t >= 5:
        return "ancilla-ancilla CNOT"
    return "ancilla-data CNOT"


def single_faults(ops):
    """Yield (site index, kind, flips, syndrome, logical class, data weight)."""
    meas = [op[1] for op in ops if op[0] == "m"]
    for i, wires in SY.fault_sites(ops):
        k = kind(ops[i] if i >= 0 else None)
        for combo in itertools.product([(0, 0)] + PAULIS, repeat=len(wires)):
            if all(p == (0, 0) for p in combo):
                continue
            x, z = [0] * SY.NQ, [0] * SY.NQ
            for w, (a, b) in zip(wires, combo):
                x[w], z[w] = a, b
            flips = {}
            for j in range(i + 1, len(ops)):
                op = ops[j]
                if op[0] == "g":
                    SY.conj(op, x, z)
                elif op[0] == "r":
                    x[op[1]] = z[op[1]] = 0
                else:
                    flips[op[1]] = x[op[1]]
            E = (tuple(x[:5]), tuple(z[:5]))
            syn = tuple(SY.sympl(E, s) for s in SY.STAB)
            log = (SY.sympl(E, XBAR), SY.sympl(E, ZBAR))
            weight = sum(1 for q in range(5) if E[0][q] or E[1][q])
            yield i, k, tuple(flips.get(a, 0) for a in meas), syn, log, weight


def analyse(name, ops):
    faults = list(single_faults(ops))
    det = Counter()
    tot = Counter()
    spread = defaultdict(Counter)
    for i, k, fl, syn, log, w in faults:
        tot[k] += 1
        detected = any(fl) or any(syn)
        det[k] += detected
        if not detected and any(log):
            raise RuntimeError("silent single fault")
        spread[k][w] += 1
    # silent pairs at distinct sites, by kind pair
    by_key = defaultdict(list)
    for f in faults:
        by_key[(f[2], f[3])].append(f)
    neg = lambda t: tuple((-v) % D for v in t)
    pairs = Counter()
    for key, group in by_key.items():
        partner = by_key.get((neg(key[0]), neg(key[1])), [])
        for a in group:
            for b in partner:
                if a[0] >= b[0]:
                    continue                       # distinct sites, each pair once
                if any((u + v) % D for u, v in zip(a[4], b[4])):
                    pairs[tuple(sorted((a[1], b[1])))] += 1
    return faults, tot, det, spread, pairs


if __name__ == "__main__":
    rows = [("current round (LRB generator)", production_ops()),
            ("current, 5 moments", R.PROD5), ("SQ14", R.SQ14), ("SQ15A", R.SQ15A)]
    for name, ops in rows:
        faults, tot, det, spread, pairs = analyse(name, ops)
        n2 = sum(1 for op in ops if op[0] == "g" and len(op) == 4)
        print(f"== {name}: {n2} two-qutrit gates, {len(faults)} single faults, 0 silent")
        for k in sorted(tot):
            sp = spread[k]
            multi = sum(c for w, c in sp.items() if w >= 2)
            print(f"   {k:22s} faults {tot[k]:5d}  detected {det[k] / tot[k]:.2f}  "
                  f"leave a data error on >= 2 qutrits {multi / tot[k]:.2f}")
        two_q = ("ancilla-data CNOT", "ancilla-ancilla CNOT", "data-data CNOT")
        tot2 = sum(c for kp, c in pairs.items() if kp[0] in two_q and kp[1] in two_q)
        print(f"   silent fault pairs: all locations {sum(pairs.values())}, "
              f"both on two-qutrit gates {tot2}")
        for kp, c in sorted(pairs.items(), key=lambda kv: -kv[1]):
            if kp[0] in two_q and kp[1] in two_q:
                print(f"      {kp[0]} + {kp[1]}: {c}")
