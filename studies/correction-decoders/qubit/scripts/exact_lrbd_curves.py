"""Exact LRB-D acceptance and accepted error at every depth, many sequences.

Uses ``lrb_qubit.exact.RawPolicyEvaluator``, so there is no shot noise; the only
randomness left is the choice of Clifford sequences. Output:
``<out>/exact_lrbd_p<p>.npz`` with arrays ``A[policy][seq, depth]`` and
``E[policy][seq, depth]`` (P(accept) and P(accept and logical flip)).

    python studies/correction-decoders/qubit/scripts/exact_lrbd_curves.py --out <dir> --workers 24
"""

from __future__ import annotations

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import exact as X  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

POLICIES = [("unif", 1), ("unif", 2), ("unif", 3), ("unif", 4), ("unif", 8),
            ("const", 1), ("const", 2), ("const", 4), ("const", 8)]


def policy_key(policy, k, final):
    return f"{policy}{k}" + ("_final" if final else "")


def run_task(args):
    p, seq_index, sequence, depths = args
    out = {}
    for d_index, depth in enumerate(depths):
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
        dec = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                  M.lrb_detector_slots(lc))
        ev = X.RawPolicyEvaluator(lc, dec)
        for policy, k in POLICIES:
            used = [lc.rounds.index(r) for r in B.used_rounds(policy, k, depth)]
            for final in (False, True):
                key = policy_key(policy, k, final)
                pa, pf = ev.evaluate(used, final)
                out.setdefault(key, np.zeros((len(depths), 2)))[d_index] = (pa,
                                                                           pf)
    return p, seq_index, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--probs", default="0.0005,0.001,0.002,0.005")
    parser.add_argument("--num-seq", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    probs = [float(x) for x in args.probs.split(",")]
    depths = list(range(args.max_depth + 1))
    sequences = E.clifford_sequences(args.num_seq, args.max_depth, 20261028)
    tasks = [(p, i, sequences[i], depths) for p in probs
             for i in range(args.num_seq)]
    results = {p: {} for p in probs}
    with ProcessPoolExecutor(args.workers) as pool:
        for p, i, out in pool.map(run_task, tasks, chunksize=2):
            for key, arr in out.items():
                results[p].setdefault(
                    key, np.zeros((args.num_seq, len(depths), 2)))[i] = arr
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    for p in probs:
        arrays = {"depths": np.array(depths)}
        for key, arr in results[p].items():
            arrays[f"A__{key}"] = arr[..., 0]
            arrays[f"E__{key}"] = arr[..., 1]
        np.savez_compressed(target / f"exact_lrbd_p{p:g}.npz", **arrays)
        print(f"p={p:g} written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
