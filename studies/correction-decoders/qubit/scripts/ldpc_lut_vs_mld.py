"""WP-C: BP+OSD, BP+LSD, union-find and sliding-window LUT decoders vs exact MLD.

For every (p, Clifford sequence, depth) the script builds the ``data5`` LRB
circuit (``circuits.build_lrb_circuit``), samples detection events, and decodes
the *same* shots with the exact ``mld.TransferMatrixMLD`` (reference) and with

* ``lut_w{w}_{joint|carry}``: ``decoders_lut.SlidingWindowLUT`` (all shots);
* the ldpc decoders of ``decoders_ldpc.build_standard_decoders`` and the
  ``mwpm_graph`` control (on a prefix of the shots holding at most
  ``--ldpc-unique`` distinct syndromes, since they decode shot by shot; the
  prefix length depends only on the syndromes, never on decoder outcomes).

Per circuit and decoder it records the number of shots, logical errors, MLD
errors on the same shots, the paired counts ``n10`` (decoder wrong, MLD right)
and ``n01``, disagreements with MLD, the posterior-weighted (Rao-Blackwell)
sums ``sum_i P(L != dec_i | s_i)`` and ``sum_{dec != mld} tanh(|LLR_i|/2)``
with their squares, and decoding times. Results are appended per task to
``<out>/raw/tasks.jsonl`` (resumable) and aggregated into small CSVs:

``summary.csv``      per (p, depth, decoder): LER, Delta-LER vs MLD (paired
                     and posterior-weighted), agreement, timing, all with SEs.
``calibration.csv``  MLD raw LER vs its posterior-predicted LER.
``lut_footprint.csv`` LUT table sizes and build times.
``per_circuit.csv.gz`` the raw per-circuit counts.

Example (login node, light):

    module load python/3.11
    python studies/correction-decoders/qubit/scripts/ldpc_lut_vs_mld.py --out /tmp/c --p-list 2e-3 \
        --depths 0,2 --num-seq 2 --shots 2000 --workers 2

Full run on the shared allocation (see results/ALLOCATION.md):

    python studies/correction-decoders/qubit/scripts/ldpc_lut_vs_mld.py \
        --out studies/correction-decoders/qubit/results/C --workers 24 \
        --max-minutes 17
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import os
import subprocess
import sys
import time
from typing import Optional

# One BLAS thread per process (see results/ALLOCATION.md): re-exec once.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import decoders_ldpc as DL  # noqa: E402
from lrb_qubit import decoders_lut as DT  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

COUNT_FIELDS = ("shots", "errors", "mld_errors", "n10", "n01", "disagree",
                "rb_err", "rb_err2", "rb_diff", "rb_diff2", "decode_seconds",
                "decode_calls", "build_seconds", "unique", "table_bits",
                "invalid_hits", "unsatisfied")


def shot_stats(pred: np.ndarray, flip: np.ndarray, mld_pred: np.ndarray,
               llr: np.ndarray) -> dict:
    """Paired and posterior-weighted counts for one decoder on given shots."""
    pred = np.asarray(pred, dtype=bool)
    wrong = pred != flip
    mld_wrong = mld_pred != flip
    disagree = pred != mld_pred
    gap = np.abs(llr)
    with np.errstate(over="ignore"):
        p_small = 1.0 / (1.0 + np.exp(gap))  # posterior of the MLD-rejected class
    # P(L != pred | s): p_small where pred agrees with MLD, 1 - p_small otherwise.
    p_wrong = np.where(disagree, 1.0 - p_small, p_small)
    diff = np.where(disagree, np.tanh(gap / 2.0), 0.0)
    return {
        "shots": int(pred.size), "errors": int(wrong.sum()),
        "mld_errors": int(mld_wrong.sum()),
        "n10": int((wrong & ~mld_wrong).sum()),
        "n01": int((~wrong & mld_wrong).sum()),
        "disagree": int(disagree.sum()),
        "rb_err": float(p_wrong.sum()), "rb_err2": float((p_wrong ** 2).sum()),
        "rb_diff": float(diff.sum()), "rb_diff2": float((diff ** 2).sum()),
    }


def ldpc_prefix(detection_events: np.ndarray, max_unique: int,
                max_shots: int) -> tuple[int, int]:
    """Longest shot prefix with at most ``max_unique`` distinct nonzero syndromes."""
    events = detection_events[:max_shots]
    rows, inverse = DL.unique_rows(events)
    nonzero = rows.any(axis=1)
    first = np.full(rows.shape[0], events.shape[0], dtype=np.int64)
    np.minimum.at(first, inverse, np.arange(events.shape[0]))
    order = np.sort(first[nonzero])
    if order.size <= max_unique:
        return events.shape[0], int(order.size)
    return int(order[max_unique]), max_unique


def run_task(task: dict, progress: Optional[str] = None) -> dict:
    p, depth, seq_index = task["p"], task["depth"], task["seq_index"]

    def mark(label: str) -> None:
        if progress:
            Path(progress).write_text(label)

    sequence = task["sequence"]
    start_task = time.perf_counter()
    lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
    dem = lc.circuit.detector_error_model()
    t0 = time.perf_counter()
    mld = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    mld_build = time.perf_counter() - t0
    sampler = lc.circuit.compile_detector_sampler(seed=task["seed"])
    dets, obs = sampler.sample(task["shots"], separate_observables=True)
    flip = obs[:, 0]
    t0 = time.perf_counter()
    llr = mld.log_likelihood_ratio(dets)
    mld_seconds = time.perf_counter() - t0
    mld_pred = llr < 0
    rows = []

    def record(name: str, family: str, pred, n: int, **extra) -> None:
        row = {"p": p, "depth": depth, "seq": seq_index, "decoder": name,
               "family": family}
        row.update(shot_stats(pred, flip[:n], mld_pred[:n], llr[:n]))
        for field in COUNT_FIELDS:
            row.setdefault(field, 0)
        row.update(extra)
        rows.append(row)

    record("mld", "mld", mld_pred, flip.size, decode_seconds=mld_seconds,
           decode_calls=flip.size, build_seconds=mld_build)

    for window in task["windows"]:
        for commit in task["commits"]:
            lut = DT.SlidingWindowLUT(dem, M.lrb_detector_slots(lc), window,
                                      commit=commit)
            t0 = time.perf_counter()
            pred = lut.predict(dets)
            seconds = time.perf_counter() - t0
            record(f"lut_w{window}_{commit}", "lut", pred, flip.size,
                   decode_seconds=seconds, decode_calls=flip.size,
                   build_seconds=lut.build_seconds,
                   table_bits=lut.table_bits(),
                   invalid_hits=lut.invalid_hits)

    if task["decoders"]:
        n_ldpc, n_unique = ldpc_prefix(dets, task["ldpc_unique"],
                                       task["ldpc_max_shots"])
        sub = dets[:n_ldpc]
        for name in task["decoders"]:
            if name in task.get("skip", ()):
                continue
            mark(name)
            t0 = time.perf_counter()
            if name == "mwpm_graph":
                graph = DL.graph_check_matrices(
                    lc.circuit.detector_error_model(decompose_errors=True))
                decoder = DL.MatchingControl(graph)
                family = "control"
            else:
                decoder = DL.build_standard_decoders(
                    lc.circuit, names=(name,), dem=dem)[name]
                family = "ldpc"
            build = time.perf_counter() - t0
            pred = decoder.predict(sub)
            record(name, family, pred, n_ldpc,
                   decode_seconds=decoder.decode_seconds,
                   decode_calls=decoder.decode_calls, build_seconds=build,
                   unique=n_unique, unsatisfied=decoder.unsatisfied)
    mark("done")
    return {"key": task["key"], "rows": rows,
            "seconds": time.perf_counter() - start_task,
            "failed_decoders": list(task.get("skip", ()))}


def run_isolated(task: dict, workdir: Path, timeout: float) -> dict:
    """Run one task in a subprocess; drop decoders that crash or hang.

    ldpc raises some errors as uncatchable C++ aborts (and its peeling
    union-find can hang), so each task runs in its own interpreter. A
    progress file names the decoder being run; if the subprocess dies or
    times out, the task is retried without that decoder, which is listed in
    ``failed_decoders``.
    """
    task = dict(task)
    task.setdefault("skip", [])
    key = task["key"]
    while True:
        # Keys contain dots (p0.002...), so never use Path.with_suffix here.
        task_file = workdir / (key + ".task.json")
        out_file = workdir / (key + ".out.json")
        progress = workdir / (key + ".progress")
        task_file.write_text(json.dumps(task))
        for f in (out_file, progress):
            if f.exists():
                f.unlink()
        try:
            proc = subprocess.run(
                [sys.executable, __file__, "--worker-task", str(task_file)],
                capture_output=True, text=True, timeout=timeout)
            ok = proc.returncode == 0 and out_file.exists()
            detail = proc.stderr[-400:]
        except subprocess.TimeoutExpired:
            ok, detail = False, "timeout"
        if ok:
            record = json.loads(out_file.read_text())
            for f in (task_file, out_file, progress):
                f.unlink(missing_ok=True)
            return record
        culprit = progress.read_text() if progress.exists() else ""
        if culprit in ("", "done") or culprit in task["skip"]:
            raise RuntimeError(f"task {task['key']} failed: {detail}")
        print(f"   {task['key']}: decoder {culprit} failed ({detail.strip()[-120:]!r}); retrying without it", flush=True)
        task["skip"] = task["skip"] + [culprit]


def first_order_task(task: dict) -> list[dict]:
    """Exact decisions on every single-mechanism syndrome (no sampling).

    For each distinct detector set ``D`` of the DEM, ``P_D(c)`` is the
    XOR-combined probability of the mechanisms with detectors ``D`` and
    observable effect ``c``. A decoder that outputs ``f(D)`` on syndrome ``D``
    fails on a single fault with probability ``P_D(1 - f(D))``; summed over
    ``D`` and divided by ``p`` this is the decoder's first-order LER
    coefficient (the slope of LER(p) as p -> 0 if single faults dominate).
    MLD's first-order floor is ``sum_D min_c P_D(c) / p``.
    """
    p, depth, seq_index = task["p"], task["depth"], task["seq_index"]
    lc = Q.build_lrb_circuit(task["sequence"], depth, p, terminal="data5")
    dem = lc.circuit.detector_error_model()
    classes: dict[tuple, list[float]] = {}
    for mech in M.dem_mechanisms(dem):
        if not mech.detectors:
            continue
        entry = classes.setdefault(mech.detectors, [0.0, 0.0])
        c = int(mech.flips_observable)
        entry[c] = DL.xor_probability(entry[c], mech.probability)
    keys = list(classes)
    probs = np.array([classes[k] for k in keys])  # (sets, 2)
    events = np.zeros((len(keys), dem.num_detectors), dtype=bool)
    for i, dets in enumerate(keys):
        events[i, list(dets)] = True
    best = probs[:, 1] > probs[:, 0]
    floor = probs.min(axis=1).sum()
    total = probs.sum()
    predictions = {}
    mld = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    predictions["mld"] = mld.log_likelihood_ratio(events) < 0
    for window in task["windows"]:
        lut = DT.SlidingWindowLUT(dem, M.lrb_detector_slots(lc), window)
        predictions[f"lut_w{window}_joint"] = lut.predict(events)
    for name in task["decoders"]:
        if name == "mwpm_graph":
            decoder = DL.MatchingControl(DL.graph_check_matrices(
                lc.circuit.detector_error_model(decompose_errors=True)))
        else:
            decoder = DL.build_standard_decoders(lc.circuit, (name,),
                                                 dem=dem)[name]
        predictions[name] = decoder.predict(events)
    rows = []
    for name, pred in predictions.items():
        wrong_mass = np.where(pred, probs[:, 0], probs[:, 1]).sum()
        off = pred != best
        rows.append({"p": p, "depth": depth, "seq": seq_index,
                     "decoder": name, "num_sets": len(keys),
                     "single_fault_mass_over_p": total / p,
                     "mld_floor_over_p": floor / p,
                     "first_order_ler_over_p": wrong_mass / p,
                     "excess_over_p": (wrong_mass - floor) / p,
                     "sets_off_first_order_argmax": int(off.sum()),
                     "mass_off_argmax_over_p": float(
                         probs[off].sum() / p)})
    return rows


def timing_benchmark(p: float, depths, sequence, decoders, windows,
                     syndromes: int = 300, shots: int = 50_000,
                     repeats: int = 3) -> list[dict]:
    """Controlled single-process timing (best of ``repeats``).

    Per-call decoders (ldpc, ``uf_dn``) are timed on the same ``syndromes``
    distinct non-zero syndromes; MLD, the LUTs and the matching control are
    timed per shot on ``shots`` vectorized shots.
    """
    out = []
    for depth in depths:
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
        dem = lc.circuit.detector_error_model()
        dets, _ = lc.circuit.compile_detector_sampler(seed=7).sample(
            shots, separate_observables=True)
        rows, _ = DL.unique_rows(dets)
        rows = rows[rows.any(axis=1)][:syndromes]

        def best(fn):
            times = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn()
                times.append(time.perf_counter() - t0)
            return min(times)

        t0 = time.perf_counter()
        mld = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
        build = time.perf_counter() - t0
        sec = best(lambda: mld.log_likelihood_ratio(dets))
        out.append({"p": p, "depth": depth, "decoder": "mld",
                    "mode": "vectorized", "n": shots,
                    "us_per_item": 1e6 * sec / shots, "build_ms": 1e3 * build})
        for window in windows:
            t0 = time.perf_counter()
            lut = DT.SlidingWindowLUT(dem, M.lrb_detector_slots(lc), window)
            build = time.perf_counter() - t0
            sec = best(lambda: lut.predict(dets))
            out.append({"p": p, "depth": depth,
                        "decoder": f"lut_w{window}_joint",
                        "mode": "vectorized", "n": shots,
                        "us_per_item": 1e6 * sec / shots,
                        "build_ms": 1e3 * build})
        for name in decoders:
            t0 = time.perf_counter()
            if name == "mwpm_graph":
                decoder = DL.MatchingControl(DL.graph_check_matrices(
                    lc.circuit.detector_error_model(decompose_errors=True)))
                build = time.perf_counter() - t0
                sec = best(lambda: decoder.predict(dets))
                out.append({"p": p, "depth": depth, "decoder": name,
                            "mode": "vectorized", "n": shots,
                            "us_per_item": 1e6 * sec / shots,
                            "build_ms": 1e3 * build})
                continue
            decoder = DL.build_standard_decoders(lc.circuit, (name,),
                                                 dem=dem)[name]
            build = time.perf_counter() - t0
            sec = best(lambda: [decoder.decode_errors(r.astype(np.uint8))
                                for r in rows])
            out.append({"p": p, "depth": depth, "decoder": name,
                        "mode": "per_syndrome", "n": int(rows.shape[0]),
                        "us_per_item": 1e6 * sec / max(rows.shape[0], 1),
                        "build_ms": 1e3 * build})
    return out


# ----------------------------------------------------------------------------
# Aggregation
# ----------------------------------------------------------------------------

def _mean_se(values: np.ndarray, variances: np.ndarray) -> tuple[float, float]:
    """Mean over sequences and its shot-noise SE given per-sequence variances."""
    return float(values.mean()), float(math.sqrt(variances.sum()) / values.size)


def aggregate(rows: list[dict]) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        groups.setdefault((row["p"], row["depth"], row["decoder"]), []).append(
            row)
    out = []
    for (p, depth, decoder), items in sorted(groups.items()):
        items = sorted(items, key=lambda r: r["seq"])
        arr = {f: np.array([r[f] for r in items], dtype=float)
               for f in COUNT_FIELDS}
        n = arr["shots"]
        ler_s = arr["errors"] / n
        mld_s = arr["mld_errors"] / n
        n10, n01 = arr["n10"], arr["n01"]
        d_s = (n10 - n01) / n
        var_d = (n10 + n01 - (n10 - n01) ** 2 / n) / n ** 2
        rb_s = arr["rb_err"] / n
        var_rb = np.maximum(arr["rb_err2"] / n - rb_s ** 2, 0.0) / n
        rbd_s = arr["rb_diff"] / n
        var_rbd = np.maximum(arr["rb_diff2"] / n - rbd_s ** 2, 0.0) / n
        dis_s = arr["disagree"] / n
        k = len(items)
        ler, ler_se_shot = _mean_se(ler_s, ler_s * (1 - ler_s) / n)
        mld_ler, _ = _mean_se(mld_s, mld_s * (1 - mld_s) / n)
        delta, delta_se = _mean_se(d_s, var_d)
        rb, rb_se = _mean_se(rb_s, var_rb)
        rbd, rbd_se = _mean_se(rbd_s, var_rbd)
        dis, dis_se = _mean_se(dis_s, dis_s * (1 - dis_s) / n)
        calls = arr["decode_calls"].sum()
        out.append({
            "p": p, "depth": depth, "decoder": decoder,
            "family": items[0]["family"], "num_seq": k,
            "shots_total": int(n.sum()), "shots_min": int(n.min()),
            "ler": ler, "ler_se_shot": ler_se_shot,
            "ler_se_seq": float(ler_s.std(ddof=1) / math.sqrt(k)) if k > 1
            else float("nan"),
            "mld_ler_same_shots": mld_ler,
            "delta_ler": delta, "delta_ler_se": delta_se,
            "delta_ler_rb": rbd, "delta_ler_rb_se": rbd_se,
            "ler_rb": rb, "ler_rb_se": rb_se,
            "agreement": 1.0 - dis, "agreement_se": dis_se,
            "n10": int(n10.sum()), "n01": int(n01.sum()),
            "us_per_shot": 1e6 * arr["decode_seconds"].sum() / n.sum(),
            "us_per_call": 1e6 * arr["decode_seconds"].sum() / max(calls, 1),
            "build_ms_mean": 1e3 * arr["build_seconds"].mean(),
            "unique_mean": float(arr["unique"].mean()),
            "table_bits": int(arr["table_bits"].max()),
            "invalid_hits": int(arr["invalid_hits"].sum()),
            "unsatisfied": int(arr["unsatisfied"].sum()),
        })
    return out


def calibration(rows: list[dict]) -> list[dict]:
    """MLD observed LER vs posterior-predicted LER (tests DEM calibration)."""
    out = []
    for summary in aggregate([r for r in rows if r["decoder"] == "mld"]):
        z = (summary["ler"] - summary["ler_rb"]) / math.sqrt(
            summary["ler_se_shot"] ** 2 + summary["ler_rb_se"] ** 2)
        out.append({"p": summary["p"], "depth": summary["depth"],
                    "shots_total": summary["shots_total"],
                    "mld_ler_observed": summary["ler"],
                    "mld_ler_observed_se": summary["ler_se_shot"],
                    "mld_ler_posterior": summary["ler_rb"],
                    "mld_ler_posterior_se": summary["ler_rb_se"],
                    "z": z})
    return out


def _write_csv(path: Path, rows: list[dict], digits: int = 7) -> None:
    if not rows:
        return
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.{digits}g}" if isinstance(v, float)
                                 else v) for k, v in row.items()})


def write_outputs(out: Path, rows: list[dict],
                  failures: Optional[list[dict]] = None) -> None:
    if failures:
        _write_csv(out / "failed_decoders.csv", failures)
    _write_csv(out / "summary.csv", aggregate(rows))
    _write_csv(out / "calibration.csv", calibration(rows))
    lut_rows = []
    for summary in aggregate([r for r in rows if r["family"] == "lut"]):
        lut_rows.append({"p": summary["p"], "depth": summary["depth"],
                         "decoder": summary["decoder"],
                         "table_bits": summary["table_bits"],
                         "table_kib": summary["table_bits"] / 8192.0,
                         "build_ms_mean": summary["build_ms_mean"],
                         "us_per_shot": summary["us_per_shot"],
                         "invalid_hits": summary["invalid_hits"]})
    _write_csv(out / "lut_footprint.csv", lut_rows)
    _write_csv(out / "per_circuit.csv.gz", sorted(
        rows, key=lambda r: (r["p"], r["depth"], r["decoder"], r["seq"])))


def load_rows(raw: Path) -> tuple[list[dict], set, list[dict]]:
    """Rows (deduplicated per circuit and decoder), done keys, failures."""
    rows, done, failures = [], set(), []
    packed = raw.with_name(raw.name + ".gz")
    if not raw.exists() and packed.exists():
        handle_ctx = gzip.open(packed, "rt")
    elif raw.exists():
        handle_ctx = open(raw)
    else:
        handle_ctx = None
    if handle_ctx is not None:
        with handle_ctx as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                done.add(record["key"])
                rows.extend(record["rows"])
                for name in record.get("failed_decoders", []):
                    failures.append({"task": record["key"], "decoder": name})
    unique, seen = [], set()
    for row in rows:
        ident = (row["p"], row["depth"], row["seq"], row["decoder"])
        if ident not in seen:
            seen.add(ident)
            unique.append(row)
    return unique, done, failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None)
    parser.add_argument("--p-list", default="5e-4,2e-3,5e-3,1e-2")
    parser.add_argument("--depths", default="0,2,6,14,22")
    parser.add_argument("--num-seq", type=int, default=10)
    parser.add_argument("--seq-seed", type=int, default=20261028)
    parser.add_argument("--shots", type=int, default=50_000)
    parser.add_argument("--ldpc-unique", type=int, default=2500)
    parser.add_argument("--ldpc-max-shots", type=int, default=50_000)
    parser.add_argument("--windows", default="1,2,3,4")
    parser.add_argument("--commits", default="joint,carry")
    parser.add_argument("--decoders",
                        default=",".join(DL.STANDARD_DECODERS + ("mwpm_graph",)))
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=14.0,
                        help="stop submitting new tasks after this wall time")
    parser.add_argument("--task-timeout", type=float, default=900.0)
    parser.add_argument("--tag", default="",
                        help="suffix for task keys (supplementary runs)")
    parser.add_argument("--only", default="",
                        help="comma-separated task-key prefixes to run "
                             "(seeds are unchanged)")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--worker-task", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--timing", action="store_true",
                        help="write timing.csv (controlled benchmark at the "
                             "first --p-list value, sequence 0)")
    parser.add_argument("--first-order", action="store_true",
                        help="write first_order.csv (exact single-fault "
                             "decisions) instead of sampling")
    args = parser.parse_args()

    if args.worker_task:
        task_file = Path(args.worker_task)
        task = json.loads(task_file.read_text())
        record = run_task(task, progress=str(task_file.with_name(
            task["key"] + ".progress")))
        task_file.with_name(task["key"] + ".out.json").write_text(
            json.dumps(record))
        return 0

    if not args.out:
        parser.error("--out is required")
    out = Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    raw = out / "raw" / "tasks.jsonl"
    rows, done, failures = load_rows(raw)
    if args.aggregate_only:
        write_outputs(out, rows, failures)
        print(f"aggregated {len(rows)} rows from {len(done)} tasks")
        return 0

    probs = [float(x) for x in args.p_list.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    sequences = E.clifford_sequences(args.num_seq, 22, args.seq_seed)
    decoders = [d for d in args.decoders.split(",") if d]
    meta = {"probs": probs, "depths": depths, "num_seq": args.num_seq,
            "seq_seed": args.seq_seed, "shots": args.shots,
            "ldpc_unique": args.ldpc_unique,
            "ldpc_max_shots": args.ldpc_max_shots,
            "windows": args.windows, "commits": args.commits,
            "decoders": decoders, "seed": args.seed, "terminal": "data5",
            "sequences": sequences}
    meta_path = out / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps(meta, indent=1))

    if args.timing:
        bench = timing_benchmark(
            probs[0], depths, sequences[0], decoders,
            [int(w) for w in args.windows.split(",") if w])
        _write_csv(out / "timing.csv", bench)
        print(f"wrote timing.csv ({len(bench)} rows)")
        return 0

    if args.first_order:
        jobs = [{"p": p, "depth": d, "seq_index": si,
                 "sequence": sequences[si],
                 "windows": [int(w) for w in args.windows.split(",") if w],
                 "decoders": decoders}
                for p in probs for d in depths for si in range(args.num_seq)]
        from concurrent.futures import ProcessPoolExecutor
        fo_rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(first_order_task, jobs):
                fo_rows.extend(result)
        summary = []
        groups: dict[tuple, list[dict]] = {}
        for row in fo_rows:
            groups.setdefault((row["p"], row["depth"], row["decoder"]),
                              []).append(row)
        for (p, depth, name), items in sorted(groups.items()):
            vals = np.array([r["first_order_ler_over_p"] for r in items])
            exc = np.array([r["excess_over_p"] for r in items])
            off = np.array([r["mass_off_argmax_over_p"] for r in items])
            k = len(items)
            sd = (lambda a: float(a.std(ddof=1) / math.sqrt(k)) if k > 1
                  else float("nan"))
            summary.append({
                "p": p, "depth": depth, "decoder": name, "num_seq": k,
                "num_sets_mean": float(np.mean([r["num_sets"]
                                                for r in items])),
                "single_fault_mass_over_p": float(np.mean(
                    [r["single_fault_mass_over_p"] for r in items])),
                "mld_floor_over_p": float(np.mean(
                    [r["mld_floor_over_p"] for r in items])),
                "first_order_ler_over_p": float(vals.mean()),
                "first_order_ler_over_p_se_seq": sd(vals),
                "excess_over_p": float(exc.mean()),
                "excess_over_p_se_seq": sd(exc),
                "mass_off_argmax_over_p": float(off.mean())})
        _write_csv(out / "first_order.csv", summary)
        print(f"wrote first_order.csv ({len(fo_rows)} rows)")
        return 0

    tasks = []
    for pi, p in enumerate(probs):
        for di, depth in enumerate(depths):
            for si in range(args.num_seq):
                base = f"p{p:g}_d{depth}_s{si}"
                key = base + args.tag
                if key in done:
                    continue
                prefixes = [x for x in args.only.split(",") if x]
                if prefixes and not any(base == x or base.startswith(x + "_")
                                        for x in prefixes):
                    continue
                tasks.append({
                    "key": key, "p": p, "depth": depth, "seq_index": si,
                    "sequence": sequences[si], "shots": args.shots,
                    "seed": args.seed + 100_000 * pi + 1_000 * si + di,
                    "windows": [int(w) for w in args.windows.split(",") if w],
                    "commits": [c for c in args.commits.split(",") if c],
                    "decoders": decoders, "ldpc_unique": args.ldpc_unique,
                    "ldpc_max_shots": args.ldpc_max_shots})
    # Expensive tasks first for load balance.
    tasks.sort(key=lambda t: -(t["depth"] + 1) * t["p"])
    print(f"{len(done)} tasks done, {len(tasks)} to run", flush=True)
    start = time.perf_counter()
    deadline = start + 60.0 * args.max_minutes
    workdir = out / "raw" / "work"
    workdir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool, \
            open(raw, "a") as sink:
        pending = list(tasks)
        running = {}
        while pending or running:
            while pending and len(running) < args.workers and \
                    time.perf_counter() < deadline:
                task = pending.pop(0)
                running[pool.submit(run_isolated, task, workdir,
                                    args.task_timeout)] = task["key"]
            if not running:
                break
            future = next(as_completed(list(running)))
            key = running.pop(future)
            try:
                record = future.result()
            except Exception as error:  # noqa: BLE001 - keep the run alive
                print(f"   {key}: FAILED ({error})", flush=True)
                continue
            sink.write(json.dumps(record) + "\n")
            sink.flush()
            failed = record.get("failed_decoders") or ""
            print(f"[{time.perf_counter() - start:7.1f}s] {record['key']} "
                  f"({record['seconds']:.1f}s), {len(pending)} pending"
                  + (f", failed: {failed}" if failed else ""), flush=True)
    rows, done, failures = load_rows(raw)
    write_outputs(out, rows, failures)
    remaining = len(pending)
    print(f"wrote outputs; {remaining} tasks not started", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
