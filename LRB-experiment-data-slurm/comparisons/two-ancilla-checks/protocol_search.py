"""Search two-ancilla check rounds on the four conventional sites.

Data q0..q4 = wires 0..4 (q2 at the centre), sites N=5 (q0,q1,q2), W=6
(q0,q2,q3), E=7 (q1,q2,q4), S=8 (q2,q3,q4). Only data-site gates between
neighbours. G3 is measured on N and G2 on E. G1 = z0 - z1 - z3 and
G4 = -x1 + x3 - x4 use a data relay: a lender site lends its part to the
shared data qutrit (q0 for G1, q4 for G4), the reader site reads the whole
check, the lender takes the part back.

Families:
- "mid": readers N and E, lenders W and S. N and E measure G3/G2, are reset
  in the middle of the round, then read G1/G4. Six readouts per round.
- "once": lenders N and E, readers W and S. N and E lend (ending back in 0)
  and measure G3/G2 in the same round with no reset in between, so every site
  is measured once per round. Four readouts per round.

Every family is a set of operations with precedence constraints. Random
topological orders are scheduled ASAP with the production scheduler
(LRBCircuitGenerator.schedule_moments); the shallowest are kept and checked
noiselessly (every readout 0, terminal G3/G4 0, logical X and Z intact).
"""
from __future__ import annotations

import random
from collections import Counter

from sdim.circuit import Circuit
from sdim.program import Program
from lrb.circuit_generator import LRBCircuitGenerator
from lrb.code_definitions import FoldedQutritDetectionCode as Code

N, W, E, S = 5, 6, 7, 8
GEN = LRBCircuitGenerator(noise_model="si1000")


def g(name, *w): return ("g", name, *w)


class Dag:
    def __init__(self):
        self.ops: dict[str, tuple] = {}
        self.before: dict[str, set[str]] = {}

    def add(self, key, op, after=()):
        self.ops[key] = op
        self.before[key] = set(after)

    def chain(self, *keys):
        for a, b in zip(keys, keys[1:]):
            self.before[b].add(a)

    def random_order(self, rng):
        done, order = set(), []
        while len(order) < len(self.ops):
            ready = [k for k in self.ops if k not in done and self.before[k] <= done]
            k = rng.choice(ready)
            done.add(k); order.append(k)
        return order


INV = {"CNOT": "CNOT_INV", "CNOT_INV": "CNOT"}


def x_relay_g4(d, lender, reader, signs):
    """G4 = -x1 + x3 - x4 through q4. lender E collects q1, or lender S
    collects q3; the reader reads q4 and the remaining qutrit."""
    far, near = (1, 3) if lender == E else (3, 1)
    a, b, c, e = signs
    d.add("g4_hl", g("H", lender)); d.add("g4_hr", g("H", reader))
    d.add("g4_col", g(a, lender, far)); d.add("g4_lend", g(b, 4, lender))
    d.add("g4_r4", g(c, reader, 4)); d.add("g4_rn", g(e, reader, near))
    d.add("g4_unl", g(INV[b], 4, lender)); d.add("g4_clr", g(INV[a], lender, far))
    d.add("g4_hil", g("H_INV", lender)); d.add("g4_hir", g("H_INV", reader))
    d.chain("g4_hl", "g4_col", "g4_lend", "g4_r4", "g4_unl", "g4_clr", "g4_hil")
    d.chain("g4_hr", "g4_r4"); d.chain("g4_hr", "g4_rn")
    d.before["g4_hir"] |= {"g4_r4", "g4_rn"}


def z_relay_g1(d, lender, reader):
    """G1 = z0 - z1 - z3 through q0."""
    if lender == W:   # W collects -z3, N reads q0 and q1
        d.add("g1_col", g("CNOT_INV", 3, W)); d.add("g1_lend", g("CNOT", W, 0))
        d.add("g1_r0", g("CNOT", 0, N)); d.add("g1_rn", g("CNOT_INV", 1, N))
        d.add("g1_unl", g("CNOT_INV", W, 0)); d.add("g1_clr", g("CNOT", 3, W))
    else:             # N collects -z1, W reads q0 and q3
        d.add("g1_col", g("CNOT_INV", 1, N)); d.add("g1_lend", g("CNOT", N, 0))
        d.add("g1_r0", g("CNOT", 0, W)); d.add("g1_rn", g("CNOT_INV", 3, W))
        d.add("g1_unl", g("CNOT_INV", N, 0)); d.add("g1_clr", g("CNOT", 1, N))
    d.chain("g1_col", "g1_lend", "g1_r0", "g1_unl", "g1_clr")


def g3_on_n(d):
    d.add("g3_h", g("H", N)); d.add("g3_0", g("CNOT", N, 0))
    d.add("g3_1", g("CNOT", N, 1)); d.add("g3_2", g("CNOT_INV", N, 2))
    d.add("g3_hi", g("H_INV", N))
    for k in ("g3_0", "g3_1", "g3_2"):
        d.chain("g3_h", k, "g3_hi")


def g2_on_e(d):
    d.add("g2_1", g("CNOT", 1, E)); d.add("g2_2", g("CNOT", 2, E))
    d.add("g2_4", g("CNOT_INV", 4, E))


def site_keys(d, site):
    return [k for k, op in d.ops.items() if op[0] == "g" and site in op[2:]]


def family(name, g4_signs, n_first, e_first):
    """n_first / e_first: which block runs first on N / E ('own' = G3/G2)."""
    d = Dag()
    for site in (N, W, E, S):
        d.add(f"r{site}", ("r", site))
    g3_on_n(d); g2_on_e(d)
    if name == "mid":
        z_relay_g1(d, lender=W, reader=N)
        x_relay_g4(d, lender=S, reader=E, signs=g4_signs)
    else:
        z_relay_g1(d, lender=N, reader=W)
        x_relay_g4(d, lender=E, reader=S, signs=g4_signs)
    own = {N: [k for k in d.ops if k.startswith("g3_")],
           E: [k for k in d.ops if k.startswith("g2_")]}
    rel = {N: [k for k in d.ops if k.startswith("g1_") and N in d.ops[k][2:]],
           E: [k for k in d.ops if k.startswith("g4_") and E in d.ops[k][2:]]}
    for site, first in ((N, n_first), (E, e_first)):
        a, b = (own[site], rel[site]) if first == "own" else (rel[site], own[site])
        if name == "mid":
            d.add(f"m{site}a", ("m", site), after=a)
            d.add(f"r{site}b", ("r", site), after=[f"m{site}a"])
            for k in b:
                d.before[k].add(f"r{site}b")
        else:
            for k in b:
                d.before[k] |= set(a)
    for site in (N, W, E, S):
        for k in site_keys(d, site):
            d.before[k].add(f"r{site}")
        mids = [k for k in d.ops if k.startswith(f"r{site}b")]
        d.add(f"m{site}", ("m", site), after=site_keys(d, site) + mids)
    return d


def to_circuit(ops, nq=9):
    c = Circuit(dimension=3, num_qudits=nq)
    for op in ops:
        if op[0] == "g":
            c.add_gate(op[1], *op[2:])
        elif op[0] == "m":
            c.add_gate("M", op[1])
        else:
            c.add_gate("RESET", op[1])
    return c


def depth(ops):
    """(gate moments, collapse moments, schedule key)."""
    moments = GEN.schedule_moments(to_circuit(ops))
    gates = sum(1 for collapse, _ in moments if not collapse)
    key = tuple(tuple(sorted(f"{o.gate_name}{GEN._operation_qudits(o)}" for o in m))
                for _, m in moments)
    return gates, len(moments) - gates, key


def run_rounds(ops, rounds=2, sandwich=False, shots=60, code=Code, prep=None):
    """Noiseless rounds; returns (all readouts 0, terminal checks 0, logical 0)."""
    c = code.logical_plus_initial_state()
    if sandwich:
        c = c + code.single_qudit_logical_gate_circuit("H")
    count, reads = Counter(), []
    for _ in range(rounds):
        for op in ops:
            if op[0] == "g":
                c.add_gate(op[1], *op[2:])
            elif op[0] == "m":
                c.add_gate("M", op[1]); reads.append((op[1], count[op[1]]))
                count[op[1]] += 1
            else:
                c.add_gate("RESET", op[1]); count[op[1]] += 1
    if sandwich:
        c = c + code.single_qudit_logical_gate_circuit("H_INV")
    final = []
    for q in range(5):
        c.add_gate("H_INV", q); c.add_gate("M", q)
        final.append((q, count[q])); count[q] += 1
    m = Program(c).simulate(shots=shots)[0]
    v = lambda w, k, s: int(m[w][k][s].measurement_value)
    ok_reads = all(v(w, k, s) == 0 for w, k in reads for s in range(shots))
    x = [[v(w, k, s) for w, k in final] for s in range(shots)]
    ok_term = all((a[0] + a[1] - a[2]) % 3 == 0 and (-a[1] + a[3] - a[4]) % 3 == 0 for a in x)
    ok_log = all((a[0] + a[3]) % 3 == 0 for a in x)
    return ok_reads and ok_term and ok_log


def correct(ops):
    return run_rounds(ops) and run_rounds(ops, sandwich=True)


def find_g4_signs(name):
    """Sign choices of the G4 relay that pass in a plain sequential order,
    with the block order on N / E that passes."""
    good = []
    for signs in __import__("itertools").product(("CNOT", "CNOT_INV"), repeat=4):
        for nf in ("own", "relay"):
            for ef in ("own", "relay"):
                d = family(name, signs, nf, ef)
                if correct([d.ops[k] for k in topo_default(d)]):
                    good.append((signs, nf, ef))
    return good


def topo_default(d):
    done, order = set(), []
    keys = list(d.ops)
    while len(order) < len(keys):
        for k in keys:
            if k not in done and d.before[k] <= done:
                done.add(k); order.append(k); break
    return order


def search(name, signs, samples=4000, seed=1, keep=12):
    rng = random.Random(seed)
    seen, results = set(), []
    for n_first in ("own", "relay"):
        for e_first in ("own", "relay"):
            d = family(name, signs, n_first, e_first)
            for _ in range(samples):
                ops = [d.ops[k] for k in d.random_order(rng)]
                gates, collapses, key = depth(ops)
                if key in seen:
                    continue
                seen.add(key)
                results.append(((gates, collapses), n_first, e_first, ops))
    results.sort(key=lambda r: (r[0][1], r[0][0]))
    verified = []
    for dep, nf, ef, ops in results:
        if len(verified) >= keep:
            break
        if correct(ops):
            verified.append((dep, nf, ef, ops))
    return verified


if __name__ == "__main__":
    for name in ("mid", "once"):
        good = find_g4_signs(name)
        print(f"{name}: G4 signs / N first / E first that pass sequentially: {good}")
        best = search(name, good[0][0])
        for dep, nf, ef, ops in best[:5]:
            print(f"  gate moments {dep[0]}, collapse moments {dep[1]}, "
                  f"N first: {nf}, E first: {ef}")
