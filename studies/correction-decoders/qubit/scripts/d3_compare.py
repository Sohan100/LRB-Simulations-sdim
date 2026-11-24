"""Work package G: distance 3 vs distance 2 for LRB with correction or detection.

Subcommands (run from the repository root after ``module load python/3.11``):

``checks``
    Structural checks of the [[13,1,3]] circuits: circuit-level distance
    (Stim hypergraph search and graphlike searches) for both measurement
    schedules, a scan over every uniform CNOT order, and single-fault
    decoding reports (which single faults each decoder gets wrong).
``run``
    Sample LRB circuits for (code, schedule, p, sequence) tasks in a process
    pool. Each task writes ``<out>/parts/<code>_<schedule>_p<p>_s<seq>.npz``
    (an ``experiment.PointResult`` for one sequence), so runs can be split
    and resumed; finished parts are skipped.
``analyze``
    Merge the parts, compute F(m) and acceptance with uncertainties, fit
    ``F = b f^m`` (the repo's weighted grid scan, offset 0), bootstrap over
    sequences, compare with exact physical RB at 5p, find pseudo-thresholds,
    and write ``summary.json``, ``tables.md`` and repo-format CSVs.

Example:

    python studies/correction-decoders/qubit/scripts/d3_compare.py run --out RESULTS --codes d3 \
        --schedules parallel --p-values 0.001 --num-seq 4 --shots 20000 \
        --decoders pm,bposd --bposd-shots 500 --workers 4
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import d3_circuits as G  # noqa: E402
from lrb_qubit import d3_code as D  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402

DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
P_VALUES = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
SEQ_SEED = 20261028  # the main grid's sequences
RB_SCALE = 5.0
CODES = {"d3": lambda: D.CODE13, "d2": G.LeadFiveQubitCode}
REPORT_POLICIES = ["lrbd_unif_1", "lrbd_unif_2", "lrbd_unif_5",
                   "lrbd_const_1", "lrbd_const_2", "lrbd_unif_1_final",
                   "lrbd_const_1_final", "const0"]


def part_path(out: Path, code: str, schedule: str, p: float, seq: int) -> Path:
    return out / "parts" / f"{code}_{schedule}_p{p:g}_s{seq:02d}.npz"


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------
def _task(args):
    (code_name, schedule, p, p_index, seq_index, sequence, depths, shots,
     decoders, decode_shots, seed, path) = args
    start = time.perf_counter()
    code = CODES[code_name]()
    result = G.run_sequence(code, schedule, p, sequence, depths, shots,
                            decoders=decoders, decode_shots=decode_shots,
                            seed=seed)
    tmp = Path(str(path) + ".tmp.npz")
    result.to_npz(str(tmp))
    os.replace(tmp, path)
    return code_name, schedule, p, seq_index, time.perf_counter() - start


def decoder_part_path(out: Path, decoder: str, code: str, schedule: str,
                      p: float, seq: int, depth: int, tag: str = "") -> Path:
    suffix = f"_{tag}" if tag else ""
    return (out / f"parts_{decoder}"
            / f"{code}_{schedule}_p{p:g}_s{seq:02d}_m{depth:02d}{suffix}.json")


def _decoder_task(args):
    (decoder, code_name, schedule, p, seq, sequence, depth, shots, seed,
     budget, path) = args
    code = CODES[code_name]()
    result = G.run_decoder_depth(code, schedule, p, sequence, depth, shots,
                                 decoder=decoder, seed=seed,
                                 time_budget=budget)
    result.update({"decoder": decoder, "code": code_name,
                   "schedule": schedule, "p": p, "seq": seq,
                   "depth": depth, "seed": seed, "time_budget": budget})
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(result))
    os.replace(tmp, path)
    return code_name, schedule, p, seq, result["seconds"]


def _parse_by_p(text: str, cast=float) -> dict[float, float]:
    out = {}
    for item in filter(None, text.split(",")):
        key, value = item.split(":")
        out[float(key)] = cast(value)
    return out


def cmd_run(args) -> int:
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    depths = [int(d) for d in args.depths.split(",")]
    p_values = [float(x) for x in args.p_values.split(",")]
    sequences = E.clifford_sequences(max(args.num_seq, 30), 22, SEQ_SEED)
    decoders_all = [d for d in args.decoders.split(",") if d]
    bposd_by_p = _parse_by_p(args.bposd_shots_by_p, int)
    if args.kind == "decoder":
        return _run_decoder_tasks(args, out, depths, p_values, sequences,
                                  bposd_by_p)
    tasks = []
    for code_name in args.codes.split(","):
        for schedule in args.schedules.split(","):
            s_index = G.SCHEDULES.index(schedule)
            for p in p_values:
                p_index = P_VALUES.index(p) if p in P_VALUES else 99
                for seq in range(args.seq_start, args.num_seq):
                    path = part_path(out, code_name, schedule, p, seq)
                    if path.exists() and not args.overwrite:
                        continue
                    decoders = [d for d in decoders_all
                                if (d != "bposd" or seq < args.bposd_seqs)
                                and (d != "mld" or code_name == "d2")]
                    decode_shots = {"bposd": bposd_by_p.get(
                                        p, args.bposd_shots),
                                    "mld": args.mld_shots}
                    seed = (args.seed + 1_000_003 * (code_name == "d2")
                            + 500_009 * s_index + 100_000 * p_index
                            + 1_000 * 997 * seq)
                    tasks.append((code_name, schedule, p, p_index, seq,
                                  sequences[seq], depths, args.shots,
                                  decoders, decode_shots, seed, path))
    # Longest tasks first: BP-OSD, high p, sequential schedule.
    tasks.sort(key=lambda t: (-("bposd" in t[8]), -t[2]))
    meta = {"started": time.strftime("%Y-%m-%d %H:%M:%S"),
            "argv": sys.argv, "depths": depths, "p_values": p_values,
            "seq_seed": SEQ_SEED, "num_tasks": len(tasks)}
    runs = out / "runs"
    runs.mkdir(exist_ok=True)
    (runs / f"run_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps(meta, indent=1))
    print(f"{len(tasks)} tasks", flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, t) for t in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            code_name, schedule, p, seq, seconds = future.result()
            print(f"[{time.perf_counter() - start:7.1f}s] {count}/{len(tasks)}"
                  f" {code_name} {schedule} p={p:g} seq={seq} "
                  f"({seconds:.1f}s)", flush=True)
    return 0


def _run_decoder_tasks(args, out, depths, p_values, sequences,
                       shots_by_p) -> int:
    """One task per (schedule, p, sequence, depth) for a slow decoder."""
    decoder = args.decoder
    (out / f"parts_{decoder}").mkdir(parents=True, exist_ok=True)
    budget_by_p = _parse_by_p(args.budget_by_p, float)
    max_depth_by_p = _parse_by_p(args.max_depth_by_p, int)
    tasks = []
    for code_name in args.codes.split(","):
        for schedule in args.schedules.split(","):
            s_index = G.SCHEDULES.index(schedule)
            for p in p_values:
                p_index = P_VALUES.index(p) if p in P_VALUES else 99
                for seq in range(args.seq_start, args.num_seq):
                    for depth in depths:
                        if depth > max_depth_by_p.get(p, 99):
                            continue
                        path = decoder_part_path(out, decoder, code_name,
                                                 schedule, p, seq, depth,
                                                 args.tag)
                        if path.exists() and not args.overwrite:
                            continue
                        seed = (args.seed + 7_777_777 + 500_009 * s_index
                                + 13_131_313 * sum(map(ord, args.tag))
                                + 100_000 * p_index + 997_000 * seq
                                + 101 * depth)
                        tasks.append((decoder, code_name, schedule, p, seq,
                                      sequences[seq], depth,
                                      int(shots_by_p.get(p, args.bposd_shots)),
                                      seed, budget_by_p.get(p, args.budget),
                                      path))
    # Priority: parallel before sequential, then mid p (pseudo-threshold
    # region) first, deep circuits first within a point.
    order = {1e-3: 0, 5e-4: 1, 2e-3: 2, 1e-4: 3, 5e-3: 4}
    tasks.sort(key=lambda t: (G.SCHEDULES.index(t[2]) == 0,
                              order.get(t[3], 9), -t[6]))
    runs = out / "runs"
    runs.mkdir(exist_ok=True)
    (runs / f"run_{decoder}_{time.strftime('%Y%m%d_%H%M%S')}.json").write_text(
        json.dumps({"argv": sys.argv, "num_tasks": len(tasks)}, indent=1))
    print(f"{len(tasks)} {decoder} tasks", flush=True)
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_decoder_task, t) for t in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            code_name, schedule, p, seq, seconds = future.result()
            if count % 25 == 0 or count == len(tasks):
                print(f"[{time.perf_counter() - start:7.1f}s] {count}/"
                      f"{len(tasks)} last {code_name} {schedule} p={p:g} "
                      f"seq={seq} ({seconds:.1f}s)", flush=True)
    return 0


# ----------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------
def _distances(lc) -> dict:
    t0 = time.perf_counter()
    hyper = lc.circuit.search_for_undetectable_logical_errors(
        dont_explore_detection_event_sets_with_size_above=4,
        dont_explore_edges_with_degree_above=4,
        dont_explore_edges_increasing_symptom_degree=False)
    seconds = time.perf_counter() - t0
    graph = lc.circuit.shortest_graphlike_error(ignore_ungraphlike_errors=True)
    try:
        decomposed = len(lc.circuit.shortest_graphlike_error(
            ignore_ungraphlike_errors=False))
    except ValueError as exc:
        decomposed = f"error: {exc}"[:200]
    return {"hypergraph_search": len(hyper), "graphlike_only": len(graph),
            "graphlike_after_decomposition": decomposed,
            "search_seconds": round(seconds, 3)}


def _order_scan(schedule: str, sequence, depth: int) -> dict:
    import itertools
    results = {}
    for z_order in itertools.permutations("NWSE"):
        for x_order in itertools.permutations("NWSE"):
            code = D.UnrotatedSurfaceCode(3, z_order=z_order, x_order=x_order)
            lc = G.build_lrb_circuit(code, sequence, depth, 1e-3,
                                     schedule=schedule)
            hyper = lc.circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=4,
                dont_explore_edges_with_degree_above=4,
                dont_explore_edges_increasing_symptom_degree=False)
            key = len(hyper)
            results[key] = results.get(key, 0) + 1
    return {str(k): v for k, v in sorted(results.items())}


def cmd_checks(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    code = D.CODE13
    grid = E.clifford_sequences(3, 22, SEQ_SEED)
    s_and_h = ["P", "H", "HP", "PH", "HPH", "P", "XZ", "H"]
    report: dict = {"code": code.describe(), "distances": [],
                    "order_scan": {}, "single_faults": []}
    for schedule in G.SCHEDULES:
        for name, seq in (("grid0", grid[0]), ("grid1", grid[1]),
                          ("S_and_H", s_and_h)):
            for depth in (1, 3, 6):
                lc = G.build_lrb_circuit(code, seq, depth, 1e-3,
                                         schedule=schedule)
                entry = {"schedule": schedule, "sequence": name,
                         "depth": depth, **_distances(lc)}
                report["distances"].append(entry)
                print(entry, flush=True)
        if not args.skip_order_scan:
            t0 = time.perf_counter()
            report["order_scan"][schedule] = _order_scan(schedule, s_and_h, 3)
            print(schedule, "order scan", report["order_scan"][schedule],
                  f"{time.perf_counter() - t0:.0f}s", flush=True)
    p = 1e-3
    for schedule in G.SCHEDULES:
        for name, seq in (("grid0", grid[0]), ("grid1", grid[1]),
                          ("only_paulis", ["X", "Z", "XZ", "I"] * 3),
                          ("only_H", ["H"] * 12), ("only_S", ["P"] * 12)):
            for depth in (2, 6):
                lc = G.build_lrb_circuit(code, seq, depth, p,
                                         schedule=schedule)
                decoders = {"pm": G.MatchingDecoder(lc.circuit),
                            "frame_matching": G.ObservableMatchingDecoder(
                                code, lc),
                            "bposd": G.BpOsdDEMDecoder(lc.circuit)}
                for dec_name, decoder in decoders.items():
                    t0 = time.perf_counter()
                    rep = G.single_fault_report(lc.circuit, decoder)
                    entry = {"schedule": schedule, "sequence": name,
                             "depth": depth, "decoder": dec_name, "p": p,
                             "num_mechanisms": rep["num_mechanisms"],
                             "num_failing": rep["num_failing"],
                             "failing_probability_over_p":
                                 rep["failing_probability"] / p,
                             "seconds": round(time.perf_counter() - t0, 2)}
                    report["single_faults"].append(entry)
                    print(entry, flush=True)
    (out / "checks.json").write_text(json.dumps(report, indent=1))
    return 0


# ----------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------
def _safe_weights(errors: np.ndarray) -> np.ndarray:
    """Copy of ``LRBPlotter._safe_weights`` in ``src/lrb/lrb_plotting.py``."""
    err = np.asarray(errors, dtype=float)
    err = np.where(np.isnan(err), np.inf, err)
    finite = err[np.isfinite(err)]
    median = float(np.median(finite)) if len(finite) else 1.0
    floor = max(median * 1e-3, 1e-12)
    err = np.where(err <= 0, floor, err)
    return 1.0 / (err * err)


def fit_decay(depths, means, stds, free_offset=False):
    """The repo's weighted f-grid scan (``_fit_decay_parameter_*_a``).

    ``F = b f^m`` (offset 0, points with F >= 1 dropped) or, with
    ``free_offset``, ``F = a + b f^m``. Returns ``(f, a, b)`` or ``None``.
    """
    x = np.asarray(depths, float)
    y = np.asarray(means, float)
    e = np.asarray(stds, float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    if not free_offset:
        ok &= y < 1.0
    x, y, e = x[ok], y[ok], e[ok]
    if len(x) < 2:
        return None
    w = _safe_weights(e)
    low, high = 1e-8, 1.0
    best = None
    for _ in range(5):
        f_grid = np.linspace(low, high, 2001)
        z = np.power(f_grid[None, :], x[:, None])
        if free_offset:
            s, sy = np.sum(w), np.sum(w * y)
            sz = np.sum(w[:, None] * z, axis=0)
            szz = np.sum(w[:, None] * z * z, axis=0)
            szy = np.sum(w[:, None] * z * y[:, None], axis=0)
            det = s * szz - sz * sz
            det = np.where(np.abs(det) < 1e-20, np.nan, det)
            a = (szz * sy - sz * szy) / det
            b = (s * szy - sz * sy) / det
        else:
            a = np.zeros(len(f_grid))
            num = np.sum(w[:, None] * z * y[:, None], axis=0)
            den = np.sum(w[:, None] * z * z, axis=0)
            b = num / np.where(den <= 0, np.nan, den)
        resid = y[:, None] - (a[None, :] + b[None, :] * z)
        chi2 = np.sum(w[:, None] * resid * resid, axis=0)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        idx = int(np.argmin(chi2))
        best = (float(f_grid[idx]), float(a[idx]), float(b[idx]))
        span = (high - low) / 10.0
        low, high = max(1e-8, best[0] - span), min(1.0, best[0] + span)
    return best


def merge_parts(out: Path):
    """Merge per-sequence parts and per-depth decoder parts into groups."""
    groups: dict[tuple[str, str, float], dict[int, E.PointResult]] = {}
    for path in sorted((out / "parts").glob("*.npz")):
        if ".tmp" in path.name:
            continue
        code, schedule, ptxt, stxt = path.stem.split("_")
        key = (code, schedule, float(ptxt[1:]))
        groups.setdefault(key, {})[int(stxt[1:])] = E.PointResult.from_npz(
            str(path))
    decoder_parts: dict[tuple[str, str, float], list[dict]] = {}
    for folder in sorted(out.glob("parts_*")):
        for path in sorted(folder.glob("*.json")):
            if ".tmp" in path.name:
                continue
            rec = json.loads(path.read_text())
            key = (rec["code"], rec["schedule"], float(rec["p"]))
            decoder_parts.setdefault(key, []).append(rec)
    merged = {}
    for key in set(groups) | set(decoder_parts):
        parts = groups.get(key, {})
        recs = decoder_parts.get(key, [])
        n_seq = max([s + 1 for s in parts] + [r["seq"] + 1 for r in recs]
                    + [1])
        depths = (next(iter(parts.values())).depths if parts else DEPTHS)
        result = E.PointResult(key[2], list(depths), n_seq)
        for seq, part in parts.items():
            for name, table in part.tables.items():
                target = result.table(name)
                target.shots[seq] = table.shots[0]
                target.accepted[seq] = table.accepted[0]
                target.errors[seq] = table.errors[0]
        for rec in recs:
            if rec["depth"] not in depths or rec["shots"] == 0:
                continue
            target = result.table(rec["decoder"])
            d_index = depths.index(rec["depth"])
            target.add(rec["seq"], d_index, rec["shots"], rec["shots"],
                       rec["errors"])
        merged[key] = result
    return merged


def series(table: B.CountTable):
    """Per-depth mean/std/SE of F and rejection over valid sequences."""
    valid_seq = table.shots.sum(axis=1) > 0
    shots = table.shots[valid_seq].astype(float)
    acc = table.accepted[valid_seq].astype(float)
    err = table.errors[valid_seq].astype(float)
    shots = np.where(shots > 0, shots, np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        fid = np.abs(1.0 - 2.0 * err / acc)
        rej = 1.0 - acc / shots
        n = np.sum(np.isfinite(fid), axis=0)
        mean = np.nanmean(fid, axis=0)
        std = np.nanstd(fid, axis=0)
        se = std / np.sqrt(np.maximum(n, 1))
        n_rej = np.sum(np.isfinite(rej), axis=0)
        rej_mean = np.nanmean(rej, axis=0)
        rej_se = np.nanstd(rej, axis=0) / np.sqrt(np.maximum(n_rej, 1))
    return {"F": mean, "F_std": std, "F_se": se, "n_seq": n,
            "rejection": rej_mean, "rejection_se": rej_se,
            "accepted": np.nansum(acc, axis=0),
            "shots": np.nansum(shots, axis=0),
            "errors": np.nansum(err, axis=0), "per_seq_F": fid,
            "per_seq_acc": acc / shots}


def fit_with_bootstrap(depths, per_seq_F, rng, n_boot=300, min_depth=0,
                       free_offset=False):
    """Repo-style fit of the sequence-mean F, bootstrap over sequences."""
    depths = np.asarray(depths)
    sel = depths >= min_depth
    F = per_seq_F[:, sel]

    def one(rows):
        sub = F[rows]
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(sub, axis=0)
            s = np.nanstd(sub, axis=0)
        return fit_decay(depths[sel], m, s, free_offset=free_offset)

    base = one(np.arange(len(F)))
    if base is None:
        return None
    samples = []
    for _ in range(n_boot):
        fit = one(rng.integers(0, len(F), len(F)))
        if fit is not None:
            samples.append(fit[0])
    samples = np.asarray(samples)
    f = base[0]
    return {"f": f, "f_se": float(np.std(samples)) if len(samples) else
            float("nan"), "b": base[2], "a": base[1],
            "r": (1 - f) / 2, "r_se": float(np.std((1 - samples) / 2))
            if len(samples) else float("nan"),
            "r_boot": ((1 - samples) / 2).tolist()}


def acceptance_decay(depths, per_seq_acc):
    """Acceptance per step ``a`` from a log-linear fit of mean acceptance."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        m = np.nanmean(per_seq_acc, axis=0)
    ok = np.isfinite(m) & (m > 0)
    if ok.sum() < 2:
        return None
    slope, _ = np.polyfit(np.asarray(depths)[ok], np.log(m[ok]), 1)
    return float(np.exp(slope))


def physical_rb(p: float, scale: float, sequences, depths, rng, n_boot=300):
    """Exact physical RB at ``p_RB = scale * p`` (no shot noise)."""
    p_rb = scale * p
    per_seq = np.array([[G.physical_rb_fidelity(s, m, p_rb) for m in depths]
                        for s in sequences])
    fit = fit_with_bootstrap(depths, per_seq, rng, n_boot, free_offset=True)
    f_ens = G.physical_rb_decay(p_rb)
    return {"p_rb": p_rb, "scale": scale, "F": per_seq.mean(axis=0).tolist(),
            "fit_seq": {k: v for k, v in fit.items() if k != "r_boot"},
            "f_ensemble": f_ens, "r_ensemble": (1 - f_ens) / 2}


def crossing(p_values, ratios):
    """Log-log interpolated p where ``ratio`` first crosses 1 upwards.

    ``inf``: ratio < 1 on the whole grid; ``0``: ratio >= 1 everywhere.
    """
    lp = np.log(p_values)
    with np.errstate(divide="ignore", invalid="ignore"):
        lr = np.log(np.asarray(ratios, float))
    for i in range(len(lp) - 1):
        if (np.isfinite(lr[i]) and np.isfinite(lr[i + 1])
                and lr[i] < 0 <= lr[i + 1]):
            t = -lr[i] / (lr[i + 1] - lr[i])
            return float(np.exp(lp[i] + t * (lp[i + 1] - lp[i])))
    finite = lr[np.isfinite(lr)]
    if len(finite) and np.all(finite < 0):
        return float("inf")
    if len(finite) and np.all(finite >= 0):
        return 0.0
    return float("nan")


def fmt(x, digits=4):
    if x is None:
        return "nan"
    try:
        if not np.isfinite(x):
            return "inf" if x == np.inf else "nan"
    except TypeError:
        return str(x)
    return f"{x:.{digits}g}"


GRID_MAIN = ROOT / "studies/correction-decoders/qubit/results/grid_main"
GRID_MAIN_INDEX = {1e-4: 1, 5e-4: 3, 1e-3: 4, 2e-3: 5, 5e-3: 6}


def grid_main_reference(rng) -> dict:
    """The lead's d=2 main grid (sequential, data5) at the WP G p values."""
    ref = {}
    for p, idx in GRID_MAIN_INDEX.items():
        path = GRID_MAIN / f"p{idx}.npz"
        if not path.exists():
            continue
        result = E.PointResult.from_npz(str(path))
        entry = {}
        for name in ("mld5", "const0_mld", "lrbd_unif_1", "lrbd_const_1",
                     "lrbd_unif_1_final"):
            if name not in result.tables:
                continue
            s = series(result.tables[name])
            fit = fit_with_bootstrap(result.depths, s["per_seq_F"], rng,
                                     n_boot=200)
            if fit:
                fit.pop("r_boot", None)
            entry[name] = {"F": s["F"].tolist(), "F_se": s["F_se"].tolist(),
                           "rejection": s["rejection"].tolist(),
                           "fit": fit,
                           "acceptance_per_step": acceptance_decay(
                               result.depths, s["per_seq_acc"])}
        ref[f"{p:g}"] = entry
    return ref


CORRECTED = ("bposd", "pm", "pmf", "mld", "const0_mld")
SCALES = (5.0, 1.0)


def cmd_analyze(args) -> int:
    out = Path(args.out)
    merged = merge_parts(out)
    rng = np.random.default_rng(12345)
    sequences = E.clifford_sequences(30, 22, SEQ_SEED)
    summary: dict = {"groups": {}, "physical_rb": {}, "pseudo_threshold": {},
                     "grid_main": grid_main_reference(rng)}
    p_seen = sorted({k[2] for k in merged})
    for scale in SCALES:
        summary["physical_rb"][f"{scale:g}p"] = {
            f"{p:g}": physical_rb(p, scale, sequences, DEPTHS, rng)
            for p in p_seen}
    csv_root = out / "csv"
    for key in sorted(merged):
        code, schedule, p = key
        result = merged[key]
        depths = result.depths
        gname = f"{code}_{schedule}_p{p:g}"
        entry = {}
        for name, table in sorted(result.tables.items()):
            s = series(table)
            if name in CORRECTED or name in REPORT_POLICIES:
                fit = fit_with_bootstrap(depths, s["per_seq_F"], rng)
                fit_late = fit_with_bootstrap(depths, s["per_seq_F"], rng,
                                              min_depth=10)
            else:  # other LRB-D policies: data and CSVs only
                fit = fit_late = None
            entry[name] = {
                "depths": depths, "F": s["F"].tolist(),
                "F_se": s["F_se"].tolist(), "n_seq": s["n_seq"].tolist(),
                "rejection": s["rejection"].tolist(),
                "rejection_se": s["rejection_se"].tolist(),
                "accepted": s["accepted"].tolist(),
                "shots": s["shots"].tolist(), "errors": s["errors"].tolist(),
                "fit": fit, "fit_m_ge_10": fit_late,
                "acceptance_per_step": acceptance_decay(depths,
                                                        s["per_seq_acc"])}
            valid = table.shots.sum(axis=1) > 0
            sub = B.CountTable(table.shots[valid], table.accepted[valid],
                               table.errors[valid])
            B.write_repo_stats(str(csv_root / gname / f"{name}.csv"), p, sub)
        summary["groups"][gname] = entry

    for scale in SCALES:
        phys = summary["physical_rb"][f"{scale:g}p"]
        for code in sorted({k[0] for k in merged}):
            for schedule in sorted({k[1] for k in merged if k[0] == code}):
                ps = [p for p in p_seen if (code, schedule, p) in merged]
                methods = set()
                for p in ps:
                    methods |= set(summary["groups"][f"{code}_{schedule}_p{p:g}"])
                for method in sorted(m for m in methods if m in CORRECTED):
                    r, r_boot = [], []
                    for p in ps:
                        g = summary["groups"][f"{code}_{schedule}_p{p:g}"]
                        fit = g.get(method, {}).get("fit")
                        r.append(fit["r"] if fit else np.nan)
                        r_boot.append(fit["r_boot"] if fit else [])
                    r_phys = np.array([phys[f"{p:g}"]["r_ensemble"]
                                       for p in ps])
                    ratio = np.array(r) / r_phys
                    p_star = crossing(np.array(ps), ratio)
                    boots = []
                    for _ in range(300):
                        rb = [rr[rng.integers(len(rr))] if rr else np.nan
                              for rr in r_boot]
                        boots.append(crossing(np.array(ps),
                                              np.array(rb) / r_phys))
                    boots = np.array(boots)
                    fin = boots[np.isfinite(boots) & (boots > 0)]
                    summary["pseudo_threshold"][
                        f"{code}_{schedule}_{method}_vs_{scale:g}p"] = {
                        "p": ps, "r": r, "r_phys": r_phys.tolist(),
                        "ratio": ratio.tolist(), "p_star": p_star,
                        "p_star_16_84": ([float(np.percentile(fin, 16)),
                                          float(np.percentile(fin, 84))]
                                         if len(fin) >= 0.5 * len(boots)
                                         else None),
                        "boot_fraction_crossing_on_grid": float(
                            len(fin) / len(boots)),
                        "boot_fraction_never_below": float(
                            np.mean(boots == 0.0)),
                        "boot_fraction_always_below": float(
                            np.mean(np.isinf(boots)))}

    for g in summary["groups"].values():
        for m in g.values():
            for k in ("fit", "fit_m_ge_10"):
                if m.get(k):
                    m[k].pop("r_boot", None)
    (out / "summary.json").write_text(json.dumps(summary, indent=1,
                                                 default=float))
    lines = ["# WP G results (generated by `d3_compare.py analyze`)", ""]
    lines += _tables(summary, p_seen)
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def _row_cells(rows, label, p_seen, key="fit"):
    cells = []
    for p in p_seen:
        m = rows[label].get(f"{p:g}")
        if not m or not m.get(key):
            cells.append("-")
        else:
            cells.append(f"{fmt(m[key]['r'], 3)} ± {fmt(m[key]['r_se'], 2)}")
    return cells


def _tables(summary, p_seen) -> list[str]:
    lines = []
    for title, key in (("all depths", "fit"), ("depths >= 10 only",
                                                "fit_m_ge_10")):
        lines.append(f"## Error per round r = (1 - f)/2, F = b f^m fit on "
                     f"{title} (± bootstrap SE over sequences)")
        lines.append("")
        lines.append("| method | " + " | ".join(f"p={p:g}" for p in p_seen)
                     + " |")
        lines.append("|---" * (len(p_seen) + 1) + "|")
        for scale in SCALES:
            phys = summary["physical_rb"][f"{scale:g}p"]
            lines.append(f"| physical RB, p_RB = {scale:g}p (exact ensemble) | "
                         + " | ".join(fmt(phys[f"{p:g}"]["r_ensemble"], 3)
                                      for p in p_seen) + " |")
        rows = {}
        for gname, g in summary["groups"].items():
            code, schedule, ptxt = gname.split("_")
            for method, m in g.items():
                if method not in CORRECTED + tuple(REPORT_POLICIES):
                    continue
                rows.setdefault(f"{code} {schedule} {method}", {})[
                    ptxt[1:]] = m
        for label in sorted(rows):
            lines.append(f"| {label} | "
                         + " | ".join(_row_cells(rows, label, p_seen, key))
                         + " |")
        if key == "fit":
            gm = summary["grid_main"]
            for name in ("mld5", "const0_mld", "lrbd_unif_1", "lrbd_const_1"):
                cells = []
                for p in p_seen:
                    m = gm.get(f"{p:g}", {}).get(name)
                    cells.append("-" if not m or not m.get("fit") else
                                 f"{fmt(m['fit']['r'], 3)} ± "
                                 f"{fmt(m['fit']['r_se'], 2)}")
                lines.append(f"| grid_main d2 sequential {name} | "
                             + " | ".join(cells) + " |")
        lines.append("")

    lines.append("## Pseudo-thresholds: p where corrected r equals physical r")
    lines.append("")
    lines.append("p* = inf: below physical on the whole grid; 0: never below. "
                 "CI = 16-84% bootstrap (shown when at least half of the "
                 "replicates cross on the grid).")
    lines.append("")
    lines.append("| code schedule decoder vs p_RB | p* | 16-84% | "
                 "P(cross) / P(never) / P(always) | r/r_phys per p |")
    lines.append("|---|---|---|---|---|")
    for key, v in sorted(summary["pseudo_threshold"].items()):
        ci = v["p_star_16_84"]
        lines.append(
            f"| {key} | {fmt(v['p_star'], 3)} | "
            f"{'-' if ci is None else fmt(ci[0], 3) + ' - ' + fmt(ci[1], 3)}"
            f" | {v['boot_fraction_crossing_on_grid']:.2f} / "
            f"{v['boot_fraction_never_below']:.2f} / "
            f"{v['boot_fraction_always_below']:.2f} | "
            + ", ".join(f"{p:g}: {fmt(r, 3)}" for p, r in
                        zip(v["p"], v["ratio"])) + " |")
    lines.append("")
    lines.append("## F(m) (± SE over sequences) and LRB-D rejection")
    lines.append("")
    lines.append("| group | method | F(0) | F(10) | F(22) | rejection(22) | "
                 "accepted(22) | acceptance/step |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for gname in sorted(summary["groups"]):
        g = summary["groups"][gname]
        for method in list(CORRECTED) + REPORT_POLICIES:
            m = g.get(method)
            if not m:
                continue
            depths = m["depths"]

            def at(arr, d, se=None):
                if d not in depths:
                    return "-"
                i = depths.index(d)
                if se is None:
                    return fmt(arr[i], 4)
                return f"{fmt(arr[i], 4)} ± {fmt(se[i], 2)}"
            acc = m.get("acceptance_per_step")
            lines.append(
                f"| {gname} | {method} | {at(m['F'], 0, m['F_se'])} | "
                f"{at(m['F'], 10, m['F_se'])} | {at(m['F'], 22, m['F_se'])} | "
                f"{at(m['rejection'], 22, m['rejection_se'])} | "
                f"{at(m['accepted'], 22)} | "
                f"{'-' if acc is None or method in CORRECTED else fmt(acc, 4)} |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--codes", default="d3")
    run.add_argument("--schedules", default="parallel,sequential")
    run.add_argument("--p-values", default=",".join(f"{p:g}" for p in P_VALUES))
    run.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    run.add_argument("--num-seq", type=int, default=30)
    run.add_argument("--seq-start", type=int, default=0)
    run.add_argument("--shots", type=int, default=200_000)
    run.add_argument("--decoders", default="pm,bposd")
    run.add_argument("--bposd-shots", type=int, default=2000)
    run.add_argument("--bposd-shots-by-p", default="",
                     help="Per-p overrides, e.g. 0.001:1000,0.005:300")
    run.add_argument("--kind", choices=("main", "decoder"), default="main",
                     help="main: LRB-D + fast decoders per sequence; "
                          "decoder: one slow decoder per (seq, depth)")
    run.add_argument("--decoder", default="bposd")
    run.add_argument("--budget", type=float, default=60.0,
                     help="Seconds per decoder task (whole chunks)")
    run.add_argument("--budget-by-p", default="")
    run.add_argument("--max-depth-by-p", default="")
    run.add_argument("--tag", default="",
                     help="Extra decoder pass: new files and seeds, summed "
                          "with earlier passes by analyze")
    run.add_argument("--bposd-seqs", type=int, default=10)
    run.add_argument("--mld-shots", type=int, default=100_000)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--overwrite", action="store_true")
    checks = sub.add_parser("checks")
    checks.add_argument("--out", required=True)
    checks.add_argument("--skip-order-scan", action="store_true")
    analyze = sub.add_parser("analyze")
    analyze.add_argument("--out", required=True)
    args = parser.parse_args()
    return {"run": cmd_run, "checks": cmd_checks,
            "analyze": cmd_analyze}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
