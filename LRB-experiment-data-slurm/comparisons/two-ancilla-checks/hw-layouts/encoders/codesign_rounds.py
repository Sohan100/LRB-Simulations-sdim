"""Build, fault-check and schedule full check rounds for co-design candidates.

A candidate gives the data placement, safe data couplers, the two ancilla sites
(wires 5 and 6) with their touched data, and which two checks each ancilla reads.
Round = reset both ancillas, half 1 (one check per ancilla), read both, half 2,
read both; no mid-round reset (ancillas are reset only at the start of a round,
as in the paper's other layouts; the pass rule "all readouts 0" is unchanged).

Protocol choice per check: minimal protocols from codesign.protocols; each check's
candidates are screened in a full round (others fixed) by single faults located in
that check's own gates, then the assembled round is fault-checked in full
(2 rounds, with and without an H-bar sandwich, as ../hw_layouts_check.fault_test).
"""
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))   # two-ancilla-checks (protocol_search, noise_bench)
sys.path.insert(0, str(HERE.parent))          # hw-layouts (hw_layouts_check)
sys.path.insert(0, str(HERE))
import codesign as CD  # noqa: E402
import hw_layouts_check as H  # noqa: E402
import protocol_search as PS  # noqa: E402

A1W, A2W = 5, 6


def check_ops(chk, anc, proto):
    kind, _ = CD.CHECKS[chk]
    ops = CD.to_ops(kind, anc, proto, chk)
    return [op for op in ops if op[0] == "g"]


def assemble(halves):
    """halves: [((chk, anc, proto), (chk, anc, proto)), (...)] -> ops, owner per op."""
    ops, owner = [("r", A1W), ("r", A2W)], [None, None]
    for half in halves:
        for chk, anc, proto in half:
            g = check_ops(chk, anc, proto)
            ops += g
            owner += [chk] * len(g)
        for chk, anc, _ in half:
            ops.append(("m", anc, chk))
            owner.append(None)
    return ops, owner


def _one(args):
    ops, f, sandwich = args
    vals = H.run(*H.build(ops, 2, fault=f, sandwich=sandwich), shots=2)
    chk, rel, dat, lg = H.classify(vals)
    return lg and not chk and not dat


def silent_faults(ops, pool):
    out = []
    flist = list(H.faults(ops))
    for sandwich in (False, True):
        res = pool.map(_one, [(ops, f, sandwich) for f in flist], chunksize=16)
        out += [f for f, s in zip(flist, res) if s]
    return out, len(flist)


def moments(ops):
    ideal = PS.to_circuit([op[:2] if op[0] == "m" else op for op in ops], 9)
    m = PS.GEN.schedule_moments(ideal)
    ng = sum(1 for c, _ in m if not c)
    return ng, len(m) - ng


def design_round(cand, pool, k_screen=8, verbose=True):
    """cand: dict(safe=frozenset pairs, t1=touch, t2=touch, r1=(chkA, chkB), r2=(chkC, chkD))."""
    safe = cand["safe"]
    touch = {A1W: cand["t1"], A2W: cand["t2"]}
    reader = {c: A1W for c in cand["r1"]}
    reader.update({c: A2W for c in cand["r2"]})
    protos = {}
    for chk, anc in reader.items():
        kind, vec = CD.CHECKS[chk]
        cost, sols = CD.protocols(kind, vec, touch[anc], safe)
        protos[chk] = sols[:k_screen]
    best = None
    orders = [[(o1[0], o2[0]), (o1[1], o2[1])] for o1, o2 in
              itertools.product(itertools.permutations(cand["r1"]), itertools.permutations(cand["r2"]))]
    # screen once (first order): per check the first protocol with no silent fault inside its own gates
    choice = {c: protos[c][0] for c in reader}
    for chk in reader:
        for p in protos[chk]:
            choice[chk] = p
            halves = [tuple((c, reader[c], choice[c]) for c in h) for h in orders[0]]
            ops, owner = assemble(halves)
            sil, _ = silent_faults(ops, pool)
            if not any(f[0] >= 0 and owner[f[0]] == chk for f in sil):
                break
    for order in orders:
        halves = [tuple((c, reader[c], choice[c]) for c in h) for h in order]
        ops, owner = assemble(halves)
        sil, nf = silent_faults(ops, pool)
        ng, nc = moments(ops)
        n2, _ = H.stats(ops)
        if verbose:
            print(f"    order {order}: {n2} two-qutrit gates, {ng} gate moments + {nc} collapse, "
                  f"silent {len(sil)}/{2 * nf}", flush=True)
        if not sil and (best is None or (ng, n2) < (best[0], best[1])):
            best = (ng, n2, nc, order, choice, ops)
    return best


def bench(ops, model, p, shots):
    import noise_bench as NB
    from lrb.code_definitions import FoldedQutritDetectionCode as C9
    ideal = PS.to_circuit([op[:2] if op[0] == "m" else op for op in ops], 9)
    cells = []
    for sandwich in (False, True):
        acc, err = NB.memory(C9, ideal, model, p, 3, sandwich, shots)
        cells.append((acc / shots, err / max(acc, 1)))
    return cells
