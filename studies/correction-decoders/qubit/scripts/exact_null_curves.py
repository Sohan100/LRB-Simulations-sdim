"""Exact null-decoder curves: no correction, no postselection.

The null decoder reports the raw terminal readout ``x0 + x3``. Its error
probability is the marginal logical-flip probability of the DEM, which the
transfer-matrix sweep gives exactly when no slot is conditioned. This is the
control that separates "correction removes non-Markovianity" from "dropping
postselection removes it".

    python studies/correction-decoders/qubit/scripts/exact_null_curves.py --out <dir> --workers 24
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
    flips = np.zeros((len(depths), 3))
    for d_index, depth in enumerate(depths):
        for col, (terminal, no_checks) in enumerate(
                (("data2", False), ("data5", False), ("data5", True))):
            if no_checks:
                lc = Q.build_lrb_const0_circuit(sequence, depth, p)
            else:
                lc = Q.build_lrb_circuit(sequence, depth, p,
                                         terminal=terminal)
            dec = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                      M.lrb_detector_slots(lc))
            _, p_flip = X.exact_condition(dec, [])
            flips[d_index, col] = p_flip
    return p, seq_index, flips


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
    flips = {p: np.zeros((args.num_seq, len(depths), 3)) for p in probs}
    with ProcessPoolExecutor(args.workers) as pool:
        for p, i, arr in pool.map(run_task, tasks, chunksize=2):
            flips[p][i] = arr
    target = Path(args.out)
    target.mkdir(parents=True, exist_ok=True)
    for p in probs:
        np.savez_compressed(target / f"exact_null_p{p:g}.npz",
                            depths=np.array(depths),
                            flip_checks_data2=flips[p][..., 0],
                            flip_checks_data5=flips[p][..., 1],
                            flip_nochecks=flips[p][..., 2])
        print(f"p={p:g} written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
