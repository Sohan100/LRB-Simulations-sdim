"""sdim verification and noisy benchmark of the one-ancilla rounds.

Rounds: the LRB generator's one-ancilla round ("production", 7 gate moments), the
same round packed into 5 gate moments (prod_reschedule.py), and the square-lattice
one-ancilla rounds chosen from oa_finalists.py (op lists in rounds.py).

verify: ../hw_layouts_check.py noiseless tests (zero noise, 5 rounds, with and
without H-bar; 40 single data Paulis: deterministic, QND, right syndrome) and the
exhaustive single-fault check (2 rounds, with and without H-bar), parallel.
bench: ../../noise_bench.memory, 3 rounds, |+_L>, production SI1000 / SD6 noise
applied by LRBCircuitGenerator exactly as in the LRB runs; accept if every readout
and the terminal G3, G4 are 0; report acceptance and logical error among accepted.

Usage: oa_verify_bench.py verify NAME... | bench MODEL P SHOTS [R=rounds] NAME...
"""
import sys
from multiprocessing import Pool
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import hw_layouts_check as H  # noqa: E402
import noise_bench as NB  # noqa: E402
import protocol_search as PS  # noqa: E402
import rounds as R  # noqa: E402
from lrb.code_definitions import FoldedQutritDetectionCode as C9  # noqa: E402


def ideal(name, model="si1000"):
    if name == "production":
        return NB.production_round("folded_qutrit", model)[0]
    ops = R.ROUNDS[name]
    return PS.to_circuit([op[:2] if op[0] == "m" else op for op in ops], 9)


def _fault(args):
    ops, f, sandwich = args
    vals = H.run(*H.build(ops, 2, fault=f, sandwich=sandwich), shots=2)
    chk, rel, dat, lg = H.classify(vals)
    return bool(lg and not chk and not dat)


def verify(name, pool):
    ops = R.ROUNDS[name]
    print(f"== {name}", flush=True)
    for ln in H.noiseless_tests(name, ops):
        print(ln, flush=True)
    flist = list(H.faults(ops))
    for sandwich in (False, True):
        res = pool.map(_fault, [(ops, f, sandwich) for f in flist], chunksize=8)
        bad = [f for f, s in zip(flist, res) if s]
        print(f"  single faults{' (H sandwich)' if sandwich else ''}: {len(flist)} faults, {len(bad)} silent logical"
              + (f"; e.g. {bad[:3]}" if bad else ""), flush=True)


def _bench(args):
    name, model, p, shots, sandwich, seed, rounds = args
    # sdim samples from NumPy's global generator and `random`; forked workers would
    # otherwise share one stream, so seed every chunk separately
    import random
    import numpy as np
    np.random.seed(seed)
    random.seed(seed)
    c = ideal(name, model)
    acc, err = NB.memory(C9, c, model, p, rounds, sandwich, shots)
    return name, model, p, sandwich, acc, err, shots


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "verify":
        with Pool(120) as pool:
            for name in sys.argv[2:]:
                verify(name, pool)
    else:
        model, p, shots = sys.argv[2], float(sys.argv[3]), int(sys.argv[4])
        names = sys.argv[5:]
        rounds = 3
        if names and names[0].startswith("R="):
            rounds = int(names[0][2:])
            names = names[1:]
        jobs = []
        chunk = 20000
        for name in names:
            for sandwich in (False, True):
                for k in range(shots // chunk):
                    jobs.append((name, model, p, chunk, sandwich, 1000 * len(jobs) + 17 + 7 * rounds, rounds))
        tot = {}
        with Pool(min(120, len(jobs))) as pool:
            for name, m, pp, sandwich, acc, err, n in pool.imap_unordered(_bench, jobs):
                a = tot.setdefault((name, sandwich), [0, 0, 0])
                a[0] += acc; a[1] += err; a[2] += n
        for name in names:
            c = ideal(name, model)
            moments = PS.GEN.schedule_moments(c)
            ng = sum(1 for col, _ in moments if not col)
            cells = []
            for sandwich in (False, True):
                acc, err, n = tot[(name, sandwich)]
                cells.append(f"acc {acc / n:.4f} err {err / max(acc, 1):.2e} ({err})")
            print(f"{model} p={p} R={rounds} shots={shots} {name:28s} {ng}g+{len(moments) - ng}c  "
                  f"X: {cells[0]}   Z: {cells[1]}", flush=True)
