"""Paired protocol-variant study for correction-based LRB.

Draws a seeded set of random Clifford sequences once and builds, for every
sequence and depth, the baseline circuit structure and several protocol
variants (see ``lrb.correction.variants``), all in memory. Stage ``prepare``
compiles every circuit into a pickled chain structure; stage ``run`` evaluates
every strategy at every requested noise strength and writes one ``.npz`` per
(variant, sequence, p). Because all variants share the same sequences, their
differences are paired comparisons.

Example (single node):

    OMP_NUM_THREADS=1 python scripts/correction/run_variant_study.py prepare \
        --cache <dir> --workers 120
    OMP_NUM_THREADS=1 python scripts/correction/run_variant_study.py run \
        --cache <dir> --out <dir> --workers 120
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lrb.correction import chain_dem, study, variants  # noqa: E402

VARIANTS = {
    "base": {},
    "term": {"full_terminal_readout": True},
    "chk": {"checked_inverse": True},
    "par": {"parallel_measurement": True},
    "par_term": {"parallel_measurement": True, "full_terminal_readout": True},
    "par_term_chk": {"parallel_measurement": True, "full_terminal_readout": True, "checked_inverse": True},
}
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]


def clifford_sequences(count: int, max_depth: int, seed: int) -> list[list[str]]:
    from lrb.code_definitions import single_qutrit_cliffords_exhaustive_strings as catalog

    rng = random.Random(seed)
    return [[rng.choice(catalog) for _ in range(max_depth)] for _ in range(count)]


def structure_path(cache: Path, variant: str, cliff: int, depth_index: int) -> Path:
    return cache / f"{variant}_c{cliff:02d}_d{depth_index}.pkl"


def prepare_task(cache: str, variant: str, cliff: int, depth_index: int, strings: list[str],
                 depths: list[int]) -> str:
    from lrb.dem_simulation import build_compact_detector_error_model

    target = structure_path(Path(cache), variant, cliff, depth_index)
    if target.exists():
        return str(target)
    depth = depths[depth_index]
    circuit = variants.build_variant_circuit(strings, depth, p=0.001, **VARIANTS[variant])
    model = build_compact_detector_error_model(circuit)
    prepared = study.PreparedCircuit(depth, chain_dem.build_chain_structure(model), None, None)
    tmp = target.with_suffix(".tmp")
    study.save_prepared(prepared, str(tmp))
    os.replace(tmp, target)
    return str(target)


def run_task(cache: str, out: str, variant: str, cliff: int, p: float, shots: int, seed: int,
             depths: list[int]) -> str:
    label = f"{p:.6g}".replace(".", "p").replace("-", "m")
    target = Path(out) / f"{variant}_c{cliff:02d}_p{label}.npz"
    if target.exists():
        return str(target)
    rng = np.random.default_rng([seed, cliff, int(round(p * 1e9)), sorted(VARIANTS).index(variant)])
    arrays: dict[str, np.ndarray] = {}
    timing = []
    for depth_index in range(len(depths)):
        start = time.perf_counter()
        prepared = study.load_prepared(str(structure_path(Path(cache), variant, cliff, depth_index)))
        result = study.evaluate_chain(prepared, p, shots, rng, windows=(0, 2), run_restricted=False)
        for key, value in result.items():
            arrays[f"d{depth_index}__{key}"] = np.asarray(value)
        timing.append(time.perf_counter() - start)
    arrays["timing_seconds"] = np.array(timing)
    arrays["p"] = np.array(p)
    arrays["cliff"] = np.array(cliff)
    tmp = target.with_name(target.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, target)
    return str(target)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["prepare", "run"])
    parser.add_argument("--cache", required=True)
    parser.add_argument("--out", default=None)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--num-cliffs", type=int, default=30)
    parser.add_argument("--sequence-seed", type=int, default=20261028)
    parser.add_argument("--exact-probs", default="2e-4,5e-4,1e-3,2e-3,5e-3")
    parser.add_argument("--sampled-probs", default="5e-4,1e-3,2e-3,5e-3")
    parser.add_argument("--shots", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--depths", default=",".join(str(d) for d in DEPTHS),
                        help="comma-separated depths, or 'dense' for 0..22")
    args = parser.parse_args()

    depths = list(range(23)) if args.depths == "dense" else [int(v) for v in args.depths.split(",")]
    names = [v for v in args.variants.split(",") if v]
    sequences = clifford_sequences(args.num_cliffs, max(DEPTHS), args.sequence_seed)
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "sequences.json").write_text(json.dumps(sequences))

    if args.stage == "prepare":
        tasks = [(str(cache), v, c, d, sequences[c], depths) for v in names
                 for c in range(args.num_cliffs) for d in range(len(depths))]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(prepare_task, *task) for task in tasks]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if count % 50 == 0 or count == len(futures):
                    print(f"prepared {count}/{len(futures)}", flush=True)
        return 0

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    exact = [float(v) for v in args.exact_probs.split(",") if v]
    sampled = [float(v) for v in args.sampled_probs.split(",") if v]
    jobs = {p: (args.shots if any(np.isclose(p, sampled)) else 0) for p in sorted(set(exact) | set(sampled))}
    (out / "variant_config.json").write_text(json.dumps({
        "variants": {v: VARIANTS[v] for v in names}, "depths": depths,
        "num_cliffs": args.num_cliffs, "sequence_seed": args.sequence_seed,
        "jobs": {repr(k): v for k, v in jobs.items()}, "seed": args.seed,
    }, indent=1))
    tasks = [(str(cache), str(out), v, c, p, shots, args.seed, depths)
             for v in names for p, shots in jobs.items() for c in range(args.num_cliffs)]
    tasks.sort(key=lambda task: -(task[5] * (2 if "chk" in task[2] else 1)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_task, *task) for task in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 20 == 0 or count == len(futures):
                print(f"finished {count}/{len(futures)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
