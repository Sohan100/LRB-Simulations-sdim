"""Run the qubit [[5,1,2]] LRB grid: LRB-D policies, exact MLD, physical RB.

Every (probability, Clifford sequence) pair is one task; tasks run in a
process pool and are merged per probability into
``<out>/p<idx>.npz`` plus repo-format CSVs under ``<out>/csv``.

Example (quick look on a login node):

    module load python/3.11
    python studies/correction-decoders/qubit/scripts/run_grid.py --out /tmp/x --p-indices 6,8 \
        --num-seq 4 --shots 20000 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# One BLAS thread per worker process; the pool provides the parallelism.
# A site hook in this Python environment imports numpy (and numba) at
# interpreter startup, so the variables must be set before the interpreter
# starts: re-exec once with them. (threadpoolctl aborts in forked workers here.)
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402

PROBS = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.0075, 0.01,
         0.0125, 0.015, 0.02, 0.03, 0.04, 0.05]
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]


def _task(args):
    p_index, p, seq_index, sequence, depths, shots, decode_shots, seed = args
    start = time.perf_counter()
    result = E.run_point(p, [sequence], depths, shots,
                         decode_shots=decode_shots,
                         seed=seed + 100_000 * p_index + 1_000_000 * seq_index)
    return p_index, seq_index, result, time.perf_counter() - start


def _merge(parts: dict[int, E.PointResult], p: float, depths, n_seq):
    merged = E.PointResult(p, list(depths), n_seq)
    for seq_index, part in parts.items():
        for name, table in part.tables.items():
            target = merged.table(name)
            target.shots[seq_index] = table.shots[0]
            target.accepted[seq_index] = table.accepted[0]
            target.errors[seq_index] = table.errors[0]
        for name, hist in part.gap_hist.items():
            merged.hist(name)[seq_index] = hist[0]
    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--p-indices", default="all")
    parser.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    parser.add_argument("--num-seq", type=int, default=30)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--decode-shots", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seq-seed", type=int, default=20261028)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--rb-shots", type=int, default=100_000)
    parser.add_argument("--rb-scale", type=float, default=5.0)
    args = parser.parse_args()

    depths = [int(d) for d in args.depths.split(",")]
    p_indices = (list(range(len(PROBS))) if args.p_indices == "all"
                 else [int(i) for i in args.p_indices.split(",")])
    sequences = E.clifford_sequences(args.num_seq, max(depths), args.seq_seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"probs": PROBS, "p_indices": p_indices, "depths": depths,
            "num_seq": args.num_seq, "shots": args.shots,
            "decode_shots": args.decode_shots, "seq_seed": args.seq_seed,
            "seed": args.seed, "sequences": sequences,
            "rb_shots": args.rb_shots, "rb_scale": args.rb_scale}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))

    tasks = [(pi, PROBS[pi], si, sequences[si], depths, args.shots,
              args.decode_shots, args.seed)
             for pi in p_indices for si in range(args.num_seq)]
    parts: dict[int, dict[int, E.PointResult]] = {pi: {} for pi in p_indices}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, t) for t in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            p_index, seq_index, result, seconds = future.result()
            parts[p_index][seq_index] = result
            if len(parts[p_index]) == args.num_seq:
                merged = _merge(parts[p_index], PROBS[p_index], depths,
                                args.num_seq)
                merged.to_npz(str(out / f"p{p_index}.npz"))
                csv_root = out / "csv" / str(p_index)
                for name, table in merged.tables.items():
                    B.write_repo_stats(str(csv_root / f"{name}.csv"),
                                       PROBS[p_index], table)
                rb = E.run_physical_rb(args.rb_scale * PROBS[p_index],
                                       sequences, depths, args.rb_shots)
                B.write_repo_stats(str(csv_root / "physical_rb.csv"),
                                   args.rb_scale * PROBS[p_index], rb)
                np.savez_compressed(out / f"rb{p_index}.npz",
                                    p_rb=args.rb_scale * PROBS[p_index],
                                    shots=rb.shots, accepted=rb.accepted,
                                    errors=rb.errors)
                print(f"[{time.perf_counter() - start:8.1f}s] p_index "
                      f"{p_index} done", flush=True)
            if count % max(1, len(tasks) // 20) == 0:
                print(f"[{time.perf_counter() - start:8.1f}s] {count}/"
                      f"{len(tasks)} tasks (last {seconds:.1f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
