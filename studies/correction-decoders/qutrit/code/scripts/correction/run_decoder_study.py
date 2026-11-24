"""Evaluate correction, detection, and hybrid decoders on baseline LRB circuits.

Stage ``prepare`` compiles every (Clifford sequence, depth) circuit of a
baseline run folder into a pickled chain structure (read-only access to the
run folder). Stage ``run`` evaluates every strategy at every requested noise
strength and writes one ``.npz`` per (Clifford sequence, p). Both stages
parallelize over tasks with a process pool; use one BLAS thread per worker.

Example (single node):

    OMP_NUM_THREADS=1 python scripts/correction/run_decoder_study.py prepare \
        --run-folder <run> --cache <dir> --workers 64
    OMP_NUM_THREADS=1 python scripts/correction/run_decoder_study.py run \
        --run-folder <run> --cache <dir> --out <dir> --workers 120 \
        --sampled-probs 1e-4,2e-4,5e-4,1e-3,2e-3,5e-3,1e-2 --shots 100000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lrb.correction import study  # noqa: E402


def read_list(path: Path, cast=float) -> list:
    text = path.read_text().strip()
    return [cast(value) for value in text.split(",") if value.strip()]


def parse_indices(text: str, limit: int) -> list[int]:
    if text == "all":
        return list(range(limit))
    out: list[int] = []
    for part in text.split(","):
        if "-" in part:
            low, high = part.split("-")
            out.extend(range(int(low), int(high) + 1))
        else:
            out.append(int(part))
    return out


def run_meta(run_folder: Path) -> dict:
    return {
        "depths": read_list(run_folder / "depths.txt", int),
        "probs": read_list(run_folder / "probs.txt", float),
        "num_cliffs": int((run_folder / "num_cliffs.txt").read_text().strip()),
        "code_name": (run_folder / "code_name.txt").read_text().strip(),
        "noise_model": (run_folder / "noise_model.txt").read_text().strip(),
    }


def structure_path(cache: Path, cliff: int, depth_index: int) -> Path:
    return cache / f"c{cliff:02d}_d{depth_index}.pkl"


def prepare_task(run_folder: str, cache: str, cliff: int, depth_index: int, depth: int, p_index: int) -> str:
    target = structure_path(Path(cache), cliff, depth_index)
    if target.exists():
        return str(target)
    experiments = Path(run_folder) / "experiments"
    lrb = experiments / "LRB" / str(cliff) / str(p_index) / f"{depth_index}.chp"
    const0 = experiments / "LRB_const0" / str(cliff) / str(p_index) / f"{depth_index}.chp"
    prepared = study.prepare_circuit(str(lrb), str(const0) if const0.exists() else None, depth)
    tmp = target.with_suffix(".tmp")
    study.save_prepared(prepared, str(tmp))
    os.replace(tmp, target)
    return str(target)


def run_task(cache: str, out: str, cliff: int, depth_indices: list[int], p: float, p_label: str,
             shots: int, seed: int) -> str:
    target = Path(out) / f"c{cliff:02d}_p{p_label}.npz"
    if target.exists():
        return str(target)
    rng = np.random.default_rng([seed, cliff, int(round(p * 1e9))])
    arrays: dict[str, np.ndarray] = {}
    timing = []
    for depth_index in depth_indices:
        start = time.perf_counter()
        prepared = study.load_prepared(str(structure_path(Path(cache), cliff, depth_index)))
        result = study.evaluate_chain(prepared, p, shots, rng)
        for key, value in result.items():
            arrays[f"d{depth_index}__{key}"] = np.asarray(value)
        timing.append(time.perf_counter() - start)
    arrays["timing_seconds"] = np.array(timing)
    arrays["p"] = np.array(p)
    arrays["cliff"] = np.array(cliff)
    arrays["depth_indices"] = np.array(depth_indices)
    tmp = target.with_name(target.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, target)
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--run-folder", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--cliffords", default="all")
    parser.add_argument("--depth-indices", default="all")
    parser.add_argument("--structure-p-index", type=int, default=4,
                        help="probability folder whose circuits are compiled (structure is p independent)")
    parser.add_argument("--exact-probs", default="all-nonzero")
    parser.add_argument("--sampled-probs", default="")
    parser.add_argument("--shots", type=int, default=100000)
    parser.add_argument("--seed", type=int, default=20261028)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    run_folder = Path(args.run_folder)
    meta = run_meta(run_folder)
    cliffs = parse_indices(args.cliffords, meta["num_cliffs"])
    depth_indices = parse_indices(args.depth_indices, len(meta["depths"]))
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)

    if args.stage == "prepare":
        tasks = [(str(run_folder), str(cache), c, d, meta["depths"][d], args.structure_p_index)
                 for c in cliffs for d in depth_indices]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(prepare_task, *task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if count % 25 == 0 or count == len(futures):
                    print(f"prepared {count}/{len(futures)}", flush=True)
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.exact_probs == "all-nonzero":
        exact_probs = [p for p in meta["probs"] if p > 0]
    else:
        exact_probs = [float(v) for v in args.exact_probs.split(",") if v]
    sampled_probs = [float(v) for v in args.sampled_probs.split(",") if v]
    jobs = {}
    for p in sorted(set(exact_probs) | set(sampled_probs)):
        jobs[p] = args.shots if any(np.isclose(p, sampled_probs)) else 0
    (out / "study_config.json").write_text(json.dumps({
        "run_folder": str(run_folder), "meta": meta, "cliffords": cliffs,
        "depth_indices": depth_indices, "jobs": {repr(k): v for k, v in jobs.items()},
        "seed": args.seed, "structure_p_index": args.structure_p_index,
    }, indent=1))
    tasks = []
    for p, shots in jobs.items():
        label = f"{p:.6g}".replace(".", "p").replace("-", "m")
        for c in cliffs:
            tasks.append((str(cache), str(out), c, depth_indices, p, label, shots, args.seed))
    # Longest (sampled) tasks first.
    tasks.sort(key=lambda task: -task[6])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, *task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 10 == 0 or count == len(futures):
                print(f"finished {count}/{len(futures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
