"""Does checking the inverse half remove the residual curvature of LRB-D unif=1?

Exact (shot-noise-free) acceptance and accepted flip probability for LRB-D
using every check round, with the repo's unchecked inverse half
(``inverse_check_every=0``) and with a check after every inverse Clifford
(``inverse_check_every=1``). Writes ``<out>/inverse_checks_p<p>.npz``.

    python studies/correction-decoders/qubit/scripts/exact_inverse_checks.py --out <dir> --workers 32
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

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import exact as X  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402


def run_task(args):
    p, seq_index, sequence, depths = args
    out = np.zeros((2, len(depths), 2))
    for col, every in enumerate((0, 1)):
        for d_index, depth in enumerate(depths):
            lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5",
                                     inverse_check_every=every)
            dec = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                      M.lrb_detector_slots(lc))
            ev = X.RawPolicyEvaluator(lc, dec)
            out[col, d_index] = ev.evaluate(range(lc.num_rounds), True)
    return p, seq_index, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--probs", default="0.001,0.005")
    parser.add_argument("--num-seq", type=int, default=100)
    parser.add_argument("--max-depth", type=int, default=40)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    probs = [float(x) for x in args.probs.split(",")]
    depths = list(range(args.max_depth + 1))
    sequences = E.clifford_sequences(args.num_seq, args.max_depth, 20261028)
    tasks = [(p, i, sequences[i], depths) for p in probs
             for i in range(args.num_seq)]
    res = {p: np.zeros((args.num_seq, 2, len(depths), 2)) for p in probs}
    with ProcessPoolExecutor(args.workers) as pool:
        for p, i, arr in pool.map(run_task, tasks, chunksize=2):
            res[p][i] = arr
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    for p in probs:
        np.savez_compressed(target / f"inverse_checks_p{p:g}.npz",
                            depths=np.array(depths),
                            A_unchecked=res[p][:, 0, :, 0],
                            E_unchecked=res[p][:, 0, :, 1],
                            A_checked=res[p][:, 1, :, 0],
                            E_checked=res[p][:, 1, :, 1])
        print(f"p={p:g} written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
