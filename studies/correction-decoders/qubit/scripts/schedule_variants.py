"""WP H: check schedules, executed-policy circuits, inverse checks, noise ablations.

Three subcommands:

``first-order``
    Exact leading-order DEM coefficients for every variant (no sampling):
    first-order MLD error ``E1`` and LRB-D rejection ``R1`` per depth, and
    their per-step slopes, in ``<out>/first_order_summary.json``.
``run``
    Samples every (variant, p, sequence) task that is not yet on disk and
    writes ``<out>/tasks/<variant>/p<p>_s<seq>.npz`` (counts per depth for
    every method; see ``lrb_qubit.variants.run_variant``). Tasks are
    independent, so the command can be re-run to resume, and ``--max-minutes``
    stops submitting new tasks after a deadline.
``analyze``
    Merges the task files and writes ``<out>/summary.json`` (F(m),
    acceptance, fits with sequence-bootstrap errors, paired differences),
    ``<out>/fm_long.csv`` (one row per variant, p, method, depth),
    ``<out>/fits_long.csv`` and ``<out>/tables.md``.

Examples (from the repository root)::

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python studies/correction-decoders/qubit/scripts/schedule_variants.py run \
        --out studies/correction-decoders/qubit/results/H \
        --groups schedules,intervals,inverse,ablations --num-seq 10 \
        --workers 24 --max-minutes 17
    python studies/correction-decoders/qubit/scripts/schedule_variants.py analyze \
        --out studies/correction-decoders/qubit/results/H --workers 4
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

# One BLAS thread per process (see API.md): re-exec before numpy is imported
# by the site hook's children.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import variants as V  # noqa: E402

PROBS = (5e-4, 2e-3, 5e-3)
DEPTHS = (0, 2, 4, 6, 10, 14, 18, 20, 22)
SEQ_SEED = 20261028
INTERVALS = (2, 3, 4, 6, 11)
CONST_KS = (2, 3, 4)
INVERSE_KS = (1, 2, 4, 11)
CHECK_SCALES = (0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0)
SCHEDULE_TAGS = {
    "serial": ("serial", True),
    "zx": ("zx", True),
    "parallel": ("parallel", True),
    "parallel-nr": ("parallel", False),
}
MAIN_METHODS = ("raw", "lrbd", "lrbd_final", "mld5", "mld2")
# Full-circuit LRB-D policies that also get acceptance / cost-adjusted fits.
EFFECTIVE_POLICIES = tuple(
    [f"lrbd_unif_{k}" for k in (1,) + INTERVALS]
    + [f"lrbd_const_{k}" for k in (1,) + CONST_KS])


def p_tag(p: float) -> str:
    return f"{p:.1e}"


def variant_list(groups: list[str]) -> list[tuple[str, V.VariantSpec, str]]:
    """``(group, spec, shot_class)``; shot_class is "full", "mid" or "decode".

    "full": ``--shots`` LRB-D shots (schedules group); "mid": ``--mid-shots``
    (interval and inverse groups); "decode": ``--decode-shots`` (ablations).
    The first ``--decode-shots`` shots of every circuit are also decoded.
    """
    out: list[tuple[str, V.VariantSpec, str]] = []
    if "schedules" in groups:
        for tag, (schedule, reset) in SCHEDULE_TAGS.items():
            out.append(("schedules", V.VariantSpec(
                f"sched-{tag}", schedule=schedule, reset=reset,
                all_policies=True), "full"))
    if "intervals" in groups:
        out.append(("intervals", V.VariantSpec("const0", const0=True), "mid"))
        for tag in ("serial", "parallel", "parallel-nr"):
            schedule, reset = SCHEDULE_TAGS[tag]
            for k in INTERVALS:
                out.append(("intervals", V.VariantSpec(
                    f"int-{tag}-unif{k}", schedule=schedule, reset=reset,
                    forward=("unif", k)), "mid"))
            out.append(("intervals", V.VariantSpec(
                f"int-{tag}-term", schedule=schedule, reset=reset,
                forward=("terminal", 0)), "mid"))
            for k in CONST_KS:
                out.append(("intervals", V.VariantSpec(
                    f"int-{tag}-const{k}", schedule=schedule, reset=reset,
                    forward=("const", k)), "mid"))
    if "inverse" in groups:
        for tag in ("serial", "parallel"):
            schedule, reset = SCHEDULE_TAGS[tag]
            for k in INVERSE_KS:
                forward = None if k == 1 else ("unif", k)
                out.append(("inverse", V.VariantSpec(
                    f"inv-{tag}-k{k}", schedule=schedule, reset=reset,
                    forward=forward, inverse_every=k), "mid"))
    if "ablations" in groups:
        # Priority order (terms named in the work package first), serial and
        # parallel interleaved, so a deadline cut drops only the extras.
        order = ("no_meas_idle", "no_cnot", "no_gate", "no_reset_idle",
                 "no_meas_flip", "no_reset_flip", "perfect_checks",
                 "no_meas_idle_data", "no_meas_idle_anc", "no_meas_depol",
                 "no_check_1q")
        assert set(order) == set(V.ABLATIONS) - {"full"}
        for name in order:
            for tag in ("serial", "parallel"):
                schedule, reset = SCHEDULE_TAGS[tag]
                out.append(("ablations", V.VariantSpec(
                    f"abl-{tag}-{name}", schedule=schedule, reset=reset,
                    ablation=name), "decode"))
    if "checkscan" in groups:
        # First-order only: check-round noise scaled by C, gate noise fixed.
        out.append(("checkscan", V.VariantSpec("const0", const0=True),
                    "decode"))
        for scale in CHECK_SCALES:
            for tag in ("serial", "parallel-nr"):
                schedule, reset = SCHEDULE_TAGS[tag]
                for k in (1, 2, 4, 11, 0):
                    forward = (None if k == 1 else ("terminal", 0) if k == 0
                               else ("unif", k))
                    label = "term" if k == 0 else f"unif{k}"
                    out.append(("checkscan", V.VariantSpec(
                        f"scan-{tag}-{label}-c{scale:g}", schedule=schedule,
                        reset=reset, forward=forward,
                        ablation=f"check_scale_{scale:g}"), "decode"))
    return out


def task_path(out: Path, spec: V.VariantSpec, p: float, seq: int) -> Path:
    return out / "tasks" / spec.name / f"p{p_tag(p)}_s{seq:02d}.npz"


def _pin_worker(queue) -> None:
    """Pool initializer: bind this worker to one CPU of the inherited set."""
    try:
        os.sched_setaffinity(0, {queue.get(timeout=5)})
    except Exception:  # noqa: BLE001 - pinning is an optimisation only
        pass


def _run_task(args):
    spec, p, seq_index, sequence, depths, shots, decode_shots, path = args
    start = time.perf_counter()
    counts = V.run_variant(spec, p, sequence, depths, shots,
                           decode_shots=decode_shots, seq_index=seq_index)
    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, depths=np.array(depths), p=np.array(p),
                        seq_index=np.array(seq_index), **counts)
    os.replace(tmp, path)
    return spec.name, p, seq_index, time.perf_counter() - start


def cmd_run(args) -> int:
    out = Path(args.out)
    groups = args.groups.split(",")
    probs = [float(x) for x in args.probs.split(",")]
    depths = [int(d) for d in args.depths.split(",")]
    seq_range = range(args.seq_start, args.num_seq)
    sequences = E.clifford_sequences(args.num_seq, max(depths), SEQ_SEED)
    tasks = []
    # Group-major order: a deadline cut leaves whole groups complete.
    for group, spec, shot_class in variant_list(groups):
        for p in probs:
            if args.only and not any(s in spec.name for s in
                                     args.only.split(",")):
                continue
            shots = {"full": args.shots, "mid": args.mid_shots,
                     "decode": args.decode_shots}[shot_class]
            for seq in seq_range:
                path = task_path(out, spec, p, seq)
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                tasks.append((spec, p, seq, sequences[seq], depths, shots,
                              args.decode_shots, path))
    meta_path = out / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"probs": list(PROBS), "depths": list(DEPTHS),
                 "seq_seed": SEQ_SEED, "shots_full": args.shots,
                 "shots_mid": args.mid_shots,
                 "decode_shots": args.decode_shots,
                 "num_seq_requested": max(args.num_seq,
                                          meta.get("num_seq_requested", 0))})
    out.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(meta, indent=1))
    print(f"{len(tasks)} tasks to run", flush=True)
    deadline = time.time() + 60.0 * args.max_minutes
    start = time.perf_counter()
    pending = list(reversed(tasks))
    done = 0
    # One CPU per worker (the inherited affinity set, e.g. from taskset);
    # unpinned workers were badly load-balanced on the shared node.
    import multiprocessing as mp
    cpu_queue = mp.get_context("fork").Queue()
    for cpu in sorted(os.sched_getaffinity(0))[:args.workers]:
        cpu_queue.put(cpu)
    with ProcessPoolExecutor(max_workers=args.workers,
                             mp_context=mp.get_context("fork"),
                             initializer=_pin_worker,
                             initargs=(cpu_queue,)) as pool:
        in_flight = set()
        while pending or in_flight:
            while (pending and len(in_flight) < 2 * args.workers
                   and time.time() < deadline):
                in_flight.add(pool.submit(_run_task, pending.pop()))
            if not in_flight:
                break
            finished, in_flight = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in finished:
                name, p, seq, seconds = future.result()
                done += 1
                if done % max(1, len(tasks) // 40) == 0 or seconds > 120:
                    print(f"[{time.perf_counter() - start:7.1f}s] {done}/"
                          f"{len(tasks)} {name} p={p:g} s={seq} "
                          f"({seconds:.1f}s)", flush=True)
    print(f"finished {done}/{len(tasks)} tasks in "
          f"{time.perf_counter() - start:.1f}s; "
          f"{len(pending)} not started", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def load_counts(out: Path, name: str, p: float, num_seq: int
                ) -> tuple[dict[str, np.ndarray], int]:
    """Stack the task files of one (variant, p) into ``(seq, depth)`` arrays."""
    files = [out / "tasks" / name / f"p{p_tag(p)}_s{s:02d}.npz"
             for s in range(num_seq)]
    files = [f for f in files if f.exists()]
    if not files:
        return {}, 0
    stacks: dict[str, list[np.ndarray]] = {}
    for f in files:
        with np.load(f) as data:
            for key in data.files:
                if key in ("depths", "p", "seq_index"):
                    continue
                stacks.setdefault(key, []).append(data[key])
    return {k: np.stack(v) for k, v in stacks.items()}, len(files)


def method_names(counts: dict[str, np.ndarray]) -> list[str]:
    return sorted({k.split("__")[0] for k in counts if k.endswith("__shots")})


def summarize_variant(job):
    """Fits and tables for one (variant, p); runs in a worker process."""
    out, name, p, num_seq, num_boot, boot_seed = job
    counts, n_seq = load_counts(Path(out), name, p, num_seq)
    if n_seq == 0:
        return name, p, None
    depths = list(DEPTHS)
    rng = np.random.default_rng(boot_seed)
    indices = V.bootstrap_indices(n_seq, num_boot, rng)
    result = {"num_seq": n_seq, "methods": {},
              "undet2_over_p": (np.mean(counts["undet2"], axis=0) / p).tolist(),
              "num_rounds": np.mean(counts["num_rounds"], axis=0).tolist()}
    boot_samples = {}
    for method in method_names(counts):
        shots = counts[f"{method}__shots"].astype(float)
        acc = counts[f"{method}__accepted"].astype(float)
        err = counts[f"{method}__errors"].astype(float)
        fid = V.per_sequence_fidelity(shots, acc, err)
        accept = acc / np.maximum(shots, 1.0)
        main = method in MAIN_METHODS
        n_boot = num_boot if main else min(num_boot, 200)
        idx = indices[:n_boot]
        est = V.fit_with_bootstrap(depths, fid, idx)
        entry = {"fit": est.as_dict()}
        boot_samples[method] = est.boot_samples
        with np.errstate(invalid="ignore"):
            f_mean, f_std = V.sequence_stats(fid)
            entry["F_mean"] = f_mean.tolist()
            entry["F_std"] = f_std.tolist()
            entry["F_sem"] = (np.nanstd(fid, axis=0, ddof=1)
                              / np.sqrt(np.sum(np.isfinite(fid), axis=0))
                              ).tolist() if n_seq > 1 else [np.nan] * len(depths)
            entry["A_mean"] = np.mean(accept, axis=0).tolist()
            entry["A_sem"] = (np.std(accept, axis=0, ddof=1) / np.sqrt(n_seq)
                              ).tolist() if n_seq > 1 else [np.nan] * len(depths)
            entry["accepted_total"] = acc.sum(axis=0).tolist()
            entry["shots_per_circuit"] = float(shots.mean())
        if method.startswith("lrbd") and (main or method in
                                          EFFECTIVE_POLICIES):
            g = np.sqrt(accept) * fid
            g_est = V.fit_with_bootstrap(depths, g, idx)
            a_est = V.fit_with_bootstrap(depths, accept, idx)
            entry["fit_effective"] = g_est.as_dict()
            entry["fit_acceptance"] = a_est.as_dict()
            boot_samples[method + "#g"] = g_est.boot_samples
        result["methods"][method] = entry
    return name, p, (result, boot_samples, indices)


def _fmt(value, err=None, digits=4):
    if value is None or not np.isfinite(value):
        return "n/a"
    if err is None or not np.isfinite(err):
        return f"{value:.{digits}f}"
    return f"{value:.{digits}f} ± {err:.{digits}f}"


def paired_difference(boot: dict, a: tuple, b: tuple) -> dict:
    """Paired bootstrap of f_a - f_b (same sequence resamples)."""
    if a not in boot or b not in boot:
        return {}
    (na, sa), (nb, sb) = boot[a], boot[b]
    if na != nb:  # different sequence sets: bootstrap indices not paired
        return {}
    n = min(len(sa), len(sb))
    diff = sa[:n] - sb[:n]
    diff = diff[np.isfinite(diff)]
    if len(diff) < 2:
        return {}
    return {"mean": float(np.mean(diff)), "std": float(np.std(diff, ddof=1)),
            "n": int(len(diff))}


def cmd_analyze(args) -> int:
    out = Path(args.out)
    probs = [float(x) for x in args.probs.split(",")]
    groups = args.groups.split(",")
    specs = [spec for _, spec, _ in variant_list(groups)]
    jobs = [(str(out), spec.name, p, args.num_seq, args.num_boot,
             1000 + i) for i, p in enumerate(probs) for spec in specs]
    # Same bootstrap indices for every variant at one p (paired comparisons):
    # the seed depends only on the p index.
    summary: dict = {"depths": list(DEPTHS), "probs": probs, "variants": {}}
    boot: dict = {}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for name, p, payload in pool.map(summarize_variant, jobs):
            if payload is None:
                continue
            result, samples, _ = payload
            summary["variants"].setdefault(name, {})[p_tag(p)] = result
            for method, s in samples.items():
                boot[(name, p_tag(p), method)] = (result["num_seq"], s)
    print(f"fits done in {time.perf_counter() - start:.1f}s", flush=True)

    # Paired differences used in the notes.
    pairs = {}
    for p in probs:
        t = p_tag(p)
        comparisons = []
        for tag in ("zx", "parallel", "parallel-nr"):
            for m in ("mld5", "mld2", "lrbd", "raw"):
                comparisons.append((f"sched-{tag}", m, "sched-serial", m))
        for tag in ("serial", "parallel", "parallel-nr"):
            full = f"sched-{tag}"
            for k in INTERVALS:
                for m in ("mld5", "mld2"):
                    comparisons.append((f"int-{tag}-unif{k}", m, full, m))
                comparisons.append((f"int-{tag}-unif{k}", "lrbd", full,
                                    f"lrbd_unif_{k}"))
                comparisons.append((f"int-{tag}-unif{k}", "lrbd#g", full,
                                    "lrbd_unif_1#g"))
            for m in ("mld5", "mld2"):
                comparisons.append((f"int-{tag}-term", m, full, m))
                comparisons.append((f"int-{tag}-term", m, "const0", "mld5"))
            comparisons.append((f"int-{tag}-term", "lrbd", full,
                                "lrbd_const_1"))
            for k in CONST_KS:
                comparisons.append((f"int-{tag}-const{k}", "lrbd", full,
                                    f"lrbd_const_{k}"))
                comparisons.append((f"int-{tag}-const{k}", "mld5", full,
                                    "mld5"))
            comparisons.append((full, "mld5", "const0", "mld5"))
        for tag in ("serial", "parallel"):
            for k in INVERSE_KS:
                base = f"sched-{tag}" if k == 1 else f"int-{tag}-unif{k}"
                for m in ("mld5", "mld2", "lrbd", "lrbd#g"):
                    comparisons.append((f"inv-{tag}-k{k}", m, base, m))
            for name in V.ABLATIONS:
                if name == "full":
                    continue
                for m in ("mld5", "mld2", "lrbd"):
                    comparisons.append((f"abl-{tag}-{name}", m,
                                        f"sched-{tag}", m))
        for va, ma, vb, mb in comparisons:
            d = paired_difference(boot, (va, t, ma), (vb, t, mb))
            if d:
                pairs[f"{va}:{ma} - {vb}:{mb} @ {t}"] = d
    summary["paired_differences"] = pairs

    # Fraction of the d=2 correction penalty removed by a schedule, where the
    # penalty is r(MLD, every round checked) - r(no-check decoding, const0):
    # (f_sched - f_serial) / (f_const0 - f_serial), paired bootstrap.
    removed = {}
    for p in probs:
        t = p_tag(p)
        for method in ("mld5", "mld2"):
            serial = boot.get(("sched-serial", t, method))
            ref = boot.get(("const0", t, "mld5"))
            if serial is None or ref is None or serial[0] != ref[0]:
                continue
            for tag in ("zx", "parallel", "parallel-nr"):
                other = boot.get((f"sched-{tag}", t, method))
                if other is None or other[0] != serial[0]:
                    continue
                n = min(len(serial[1]), len(ref[1]), len(other[1]))
                with np.errstate(invalid="ignore", divide="ignore"):
                    frac = ((other[1][:n] - serial[1][:n])
                            / (ref[1][:n] - serial[1][:n]))
                frac = frac[np.isfinite(frac)]
                fits = summary["variants"]
                point = ((fits[f"sched-{tag}"][t]["methods"][method]["fit"]["f"]
                          - fits["sched-serial"][t]["methods"][method]["fit"]["f"])
                         / (fits["const0"][t]["methods"]["mld5"]["fit"]["f"]
                            - fits["sched-serial"][t]["methods"][method]["fit"]["f"]))
                removed[f"{tag}:{method} @ {t}"] = {
                    "fraction": float(point),
                    "std": float(np.std(frac, ddof=1)) if len(frac) > 1
                    else float("nan")}
    summary["penalty_removed"] = removed

    (out / "summary.json").write_text(json.dumps(summary, indent=1,
                                                 default=float))
    write_long_csvs(out, summary)
    write_tables(out, summary)
    print(f"wrote {out / 'summary.json'}", flush=True)
    return 0


def write_long_csvs(out: Path, summary: dict) -> None:
    depths = summary["depths"]
    with open(out / "fm_long.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "p", "method", "depth", "F_mean", "F_sem",
                    "F_std", "A_mean", "A_sem", "accepted_total", "num_seq"])
        for name, by_p in sorted(summary["variants"].items()):
            for t, res in sorted(by_p.items()):
                for method, e in sorted(res["methods"].items()):
                    for i, d in enumerate(depths):
                        w.writerow([name, t, method, d, e["F_mean"][i],
                                    e["F_sem"][i], e["F_std"][i],
                                    e["A_mean"][i], e["A_sem"][i],
                                    e["accepted_total"][i], res["num_seq"]])
    with open(out / "fits_long.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["variant", "p", "method", "quantity", "f", "f_err",
                    "f_lo", "f_hi", "b", "n_points", "f_boot_valid",
                    "num_seq"])
        for name, by_p in sorted(summary["variants"].items()):
            for t, res in sorted(by_p.items()):
                for method, e in sorted(res["methods"].items()):
                    for q in ("fit", "fit_effective", "fit_acceptance"):
                        if q in e:
                            fit = e[q]
                            w.writerow([name, t, method, q, fit["f"],
                                        fit["f_err"], fit["f_lo"],
                                        fit["f_hi"], fit["b"],
                                        fit["n_points"], fit["f_boot_valid"],
                                        res["num_seq"]])


def _get(summary, name, t, method, q="fit"):
    try:
        return summary["variants"][name][t]["methods"][method][q]
    except KeyError:
        return None


def _rate_cell(summary, name, t, method, p, q="fit"):
    """``r/p = (1-f)/(2p)`` with its bootstrap error."""
    fit = _get(summary, name, t, method, q)
    if fit is None or not np.isfinite(fit["f"]):
        return "n/a"
    r = (1 - fit["f"]) / (2 * p)
    e = fit["f_err"] / (2 * p) if np.isfinite(fit["f_err"]) else np.nan
    return _fmt(r, e, 2)


def _f_cell(summary, name, t, method, q="fit"):
    fit = _get(summary, name, t, method, q)
    if fit is None:
        return "n/a"
    return _fmt(fit["f"], fit["f_err"], 4)


def _at(summary, name, t, method, key, depth_index=-1, digits=3):
    try:
        e = summary["variants"][name][t]["methods"][method]
    except KeyError:
        return "n/a"
    v = e[key][depth_index]
    s = e[key.replace("mean", "sem")][depth_index]
    if not np.isfinite(v):
        return "n/a"
    if key.startswith("A") and v < 1e-2:
        return f"{v:.2e} ± {s:.1e}" if np.isfinite(s) else f"{v:.2e}"
    return _fmt(v, s, digits)


def write_tables(out: Path, summary: dict) -> None:
    lines = ["# WP H tables (generated by schedule_variants.py analyze)", ""]
    probs = summary["probs"]
    depths = summary["depths"]
    for p in probs:
        t = p_tag(p)
        lines += [f"## p = {p:g}", ""]
        lines += ["### Schedules (all check rounds executed)", "",
                  "| schedule | MLD5 f | MLD5 r/p | MLD2 r/p | LRB-D unif1 f | "
                  "unif1 A(22) | unif1 rej/round | raw r/p | data2 floor/p |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for tag in SCHEDULE_TAGS:
            n = f"sched-{tag}"
            acc_fit = _get(summary, n, t, "lrbd", "fit_acceptance")
            rej = ("n/a" if acc_fit is None else
                   _fmt(1 - acc_fit["f"], acc_fit["f_err"], 4))
            floor = summary["variants"].get(n, {}).get(t, {}).get(
                "undet2_over_p", [np.nan])[-1]
            lines.append(
                f"| {tag} | {_f_cell(summary, n, t, 'mld5')} | "
                f"{_rate_cell(summary, n, t, 'mld5', p)} | "
                f"{_rate_cell(summary, n, t, 'mld2', p)} | "
                f"{_f_cell(summary, n, t, 'lrbd')} | "
                f"{_at(summary, n, t, 'lrbd', 'A_mean')} | {rej} | "
                f"{_rate_cell(summary, n, t, 'raw', p)} | {floor:.2f} |")
        lines += ["", "### Check interval (forward rounds executed only where "
                  "unif=k / const=k checks)", "",
                  "| schedule | variant | MLD5 r/p | MLD2 r/p | LRB-D f | "
                  "A(22) | LRB-D g (sqrt(A) F) |",
                  "|---|---|---|---|---|---|---|"]
        rows = [("all", "sched-{tag}")]
        rows += [(f"unif{k}", f"int-{{tag}}-unif{k}") for k in INTERVALS]
        rows += [("terminal only", "int-{tag}-term")]
        rows += [(f"const{k}", f"int-{{tag}}-const{k}") for k in CONST_KS]
        for tag in ("serial", "parallel", "parallel-nr"):
            for label, pattern in rows:
                n = pattern.format(tag=tag)
                lines.append(
                    f"| {tag} | {label} | "
                    f"{_rate_cell(summary, n, t, 'mld5', p)} | "
                    f"{_rate_cell(summary, n, t, 'mld2', p)} | "
                    f"{_f_cell(summary, n, t, 'lrbd')} | "
                    f"{_at(summary, n, t, 'lrbd', 'A_mean')} | "
                    f"{_f_cell(summary, n, t, 'lrbd', 'fit_effective')} |")
        lines.append(
            f"| none | const0 | {_rate_cell(summary, 'const0', t, 'mld5', p)}"
            f" | n/a | {_f_cell(summary, 'const0', t, 'lrbd')} | "
            f"{_at(summary, 'const0', t, 'lrbd', 'A_mean')} | "
            f"{_f_cell(summary, 'const0', t, 'lrbd', 'fit_effective')} |")
        lines += ["", "### LRB-D: policy on the full repo circuit vs only "
                  "its rounds executed", "",
                  "| schedule | policy | full-circuit f | full A(22) | "
                  "full g | executed f | executed A(22) | executed g |",
                  "|---|---|---|---|---|---|---|---|"]
        policies = [(f"unif={k}", f"lrbd_unif_{k}", f"int-{{tag}}-unif{k}")
                    for k in INTERVALS]
        policies += [(f"const={k}", f"lrbd_const_{k}",
                      f"int-{{tag}}-const{k}") for k in CONST_KS]
        policies += [("const=1 (terminal)", "lrbd_const_1", "int-{tag}-term")]
        for tag in ("serial", "parallel", "parallel-nr"):
            full = f"sched-{tag}"
            for label, method, pattern in policies:
                n = pattern.format(tag=tag)
                lines.append(
                    f"| {tag} | {label} | {_f_cell(summary, full, t, method)}"
                    f" | {_at(summary, full, t, method, 'A_mean')} | "
                    f"{_f_cell(summary, full, t, method, 'fit_effective')} | "
                    f"{_f_cell(summary, n, t, 'lrbd')} | "
                    f"{_at(summary, n, t, 'lrbd', 'A_mean')} | "
                    f"{_f_cell(summary, n, t, 'lrbd', 'fit_effective')} |")
        lines += ["", "### Inverse-half checks (forward unif=k plus inverse "
                  "every k)", "",
                  "| schedule | k | MLD5 r/p | MLD2 r/p | LRB-D f | A(22) | "
                  "LRB-D g |", "|---|---|---|---|---|---|---|"]
        for tag in ("serial", "parallel"):
            for k in INVERSE_KS:
                n = f"inv-{tag}-k{k}"
                lines.append(
                    f"| {tag} | {k} | {_rate_cell(summary, n, t, 'mld5', p)}"
                    f" | {_rate_cell(summary, n, t, 'mld2', p)} | "
                    f"{_f_cell(summary, n, t, 'lrbd')} | "
                    f"{_at(summary, n, t, 'lrbd', 'A_mean')} | "
                    f"{_f_cell(summary, n, t, 'lrbd', 'fit_effective')} |")
        lines += ["", "### Ablations (all rounds executed; r/p of MLD5 and "
                  "paired change vs full)", "",
                  "| schedule | ablation | MLD5 r/p | Δ(r/p) vs full | "
                  "MLD2 r/p | LRB-D unif1 rej/round |",
                  "|---|---|---|---|---|---|"]
        pairs = summary.get("paired_differences", {})
        for tag in ("serial", "parallel"):
            for name in V.ABLATIONS:
                n = f"sched-{tag}" if name == "full" else f"abl-{tag}-{name}"
                key = f"{n}:mld5 - sched-{tag}:mld5 @ {t}"
                d = pairs.get(key)
                # f_abl - f_full > 0 means the ablation lowers r by that/2.
                delta = ("0" if name == "full" else "n/a" if not d else
                         _fmt(-d["mean"] / (2 * p), d["std"] / (2 * p), 2))
                acc_fit = _get(summary, n, t, "lrbd", "fit_acceptance")
                rej = ("n/a" if acc_fit is None else
                       _fmt(1 - acc_fit["f"], acc_fit["f_err"], 4))
                lines.append(
                    f"| {tag} | {name} | "
                    f"{_rate_cell(summary, n, t, 'mld5', p)} | {delta} | "
                    f"{_rate_cell(summary, n, t, 'mld2', p)} | {rej} |")
        lines += ["", "### F(m) of key variants (mean ± s.e.m. over "
                  "sequences)", "",
                  "| variant | method | " + " | ".join(
                      f"m={d}" for d in depths) + " |",
                  "|---|---|" + "---|" * len(depths)]
        key_rows = [("sched-serial", "mld5"), ("sched-parallel", "mld5"),
                    ("sched-parallel-nr", "mld5"), ("const0", "mld5"),
                    ("int-parallel-term", "mld5"),
                    ("sched-serial", "lrbd"), ("sched-parallel", "lrbd"),
                    ("sched-parallel-nr", "lrbd"), ("const0", "lrbd"),
                    ("sched-serial", "raw")]
        for n, m in key_rows:
            cells = [_at(summary, n, t, m, "F_mean", i) for i in
                     range(len(depths))]
            lines.append(f"| {n} | {m} | " + " | ".join(cells) + " |")
        lines += ["", "Acceptance A(m):", "",
                  "| variant | method | " + " | ".join(
                      f"m={d}" for d in depths) + " |",
                  "|---|---|" + "---|" * len(depths)]
        for n, m in key_rows:
            if not m.startswith("lrbd"):
                continue
            cells = [_at(summary, n, t, m, "A_mean", i) for i in
                     range(len(depths))]
            lines.append(f"| {n} | {m} | " + " | ".join(cells) + " |")
        lines.append("")
    removed = summary.get("penalty_removed", {})
    if removed:
        lines += ["## Fraction of the d=2 correction penalty removed", "",
                  "Penalty = r(MLD, every round checked, serial) - "
                  "r(no-check decoding, const0); fraction = (f_sched - "
                  "f_serial) / (f_const0 - f_serial), paired bootstrap.", "",
                  "| schedule | decoder | " + " | ".join(
                      f"p={p:g}" for p in probs) + " |",
                  "|---|---|" + "---|" * len(probs)]
        for tag in ("zx", "parallel", "parallel-nr"):
            for method in ("mld5", "mld2"):
                cells = []
                for p in probs:
                    e = removed.get(f"{tag}:{method} @ {p_tag(p)}")
                    cells.append("n/a" if e is None else
                                 _fmt(e["fraction"], e["std"], 3))
                lines.append(f"| {tag} | {method} | " + " | ".join(cells)
                             + " |")
        lines.append("")
    (out / "tables.md").write_text("\n".join(lines))


P_SMALL = 1e-6
FIRST_ORDER_KEYS = ("E1_5", "E1_2", "R1_5", "R1_2", "A2_5", "U1_2")


def _first_order_job(job):
    spec, seq_index, sequence, depths = job
    rows = []
    for depth in depths:
        lc = spec.build(sequence, depth, P_SMALL, terminal="data5")
        coeffs = V.first_order_coefficients(lc, P_SMALL)
        rows.append({"variant": spec.name, "seq": seq_index, "depth": depth,
                     **coeffs})
    return rows


def cmd_first_order(args) -> int:
    """Exact first-order DEM coefficients for every variant (no sampling)."""
    out = Path(args.out)
    depths = list(DEPTHS)
    sequences = E.clifford_sequences(args.num_seq, max(depths), SEQ_SEED)
    specs = [spec for _, spec, _ in variant_list(args.groups.split(","))]
    jobs = [(spec, s, sequences[s], depths) for spec in specs
            for s in range(args.num_seq)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(_first_order_job, jobs, chunksize=4):
            rows.extend(part)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out / f"first_order{args.tag}_rows.csv", "w",
              newline="") as fh:
        w = csv.writer(fh)
        w.writerow(keys)
        for row in rows:
            w.writerow([row[k] for k in keys])
    # Per-sequence least-squares slope over depth, then mean and s.e.m.
    x = np.asarray(depths, dtype=float)
    summary = {"p_small": P_SMALL, "depths": depths,
               "num_seq": args.num_seq, "variants": {}}
    for spec in specs:
        sel = [r for r in rows if r["variant"] == spec.name]
        entry = {}
        for key in FIRST_ORDER_KEYS:
            table = np.zeros((args.num_seq, len(depths)))
            for r in sel:
                table[r["seq"], depths.index(r["depth"])] = r[key]
            slopes = np.polyfit(x, table.T, 1)[0]
            entry[key] = {
                "mean_by_depth": table.mean(axis=0).tolist(),
                "slope_mean": float(slopes.mean()),
                "slope_sem": float(slopes.std(ddof=1) / np.sqrt(len(slopes)))
                if len(slopes) > 1 else float("nan")}
        summary["variants"][spec.name] = entry
    (out / f"first_order{args.tag}_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(f"{'variant':32s} " + " ".join(f"{k + '/step':>14s}"
                                         for k in FIRST_ORDER_KEYS[:4]))
    for spec in specs:
        e = summary["variants"][spec.name]
        print(f"{spec.name:32s} " + " ".join(
            f"{e[k]['slope_mean']:8.2f}±{e[k]['slope_sem']:5.2f}"
            for k in FIRST_ORDER_KEYS[:4]))
    return 0


def _fo_slopes(rows: list[dict], variant: str, key: str, depths, n_seq):
    table = np.full((n_seq, len(depths)), np.nan)
    for r in rows:
        if r["variant"] == variant:
            table[int(r["seq"]), depths.index(int(r["depth"]))] = float(r[key])
    if np.isnan(table).any():
        return None
    return np.polyfit(np.asarray(depths, float), table.T, 1)[0]


def _ms(values) -> str:
    if values is None:
        return "n/a"
    sem = values.std(ddof=1) / np.sqrt(len(values)) if len(values) > 1 else 0
    return f"{values.mean():.2f} ± {sem:.2f}"


def cmd_first_order_tables(args) -> int:
    """Markdown tables from first_order*_rows.csv (paired per sequence)."""
    out = Path(args.out)
    depths = list(DEPTHS)
    rows = []
    for name in ("first_order_rows.csv", "first_order_checkscan_rows.csv"):
        path = out / name
        if path.exists():
            with open(path) as fh:
                rows.extend(csv.DictReader(fh))
    n_seq = 1 + max(int(r["seq"]) for r in rows)

    def sl(v, k="E1_5"):
        return _fo_slopes(rows, v, k, depths, n_seq)

    def diff(a, b, k="E1_5"):
        sa, sb = sl(a, k), sl(b, k)
        return None if sa is None or sb is None else sa - sb

    lines = ["# WP H exact first-order tables (p -> 0, DEM coefficients)", "",
             f"Slopes per depth step, mean ± s.e.m. over {n_seq} sequences. "
             "E1: first-order MLD error / p (mld5 uses the final data "
             "detectors, mld2 does not). R1: first-order LRB-D rejection / p "
             "(all executed rounds + final data check). A2: second-order "
             "accepted LRB-D error / p^2.", "",
             "## Schedules, every forward round checked", "",
             "| variant | E1 mld5 | E1 mld2 | R1 | A2 | Δ E1 vs serial |",
             "|---|---|---|---|---|---|"]
    for v in ("sched-serial", "sched-zx", "sched-parallel",
              "sched-parallel-nr", "const0"):
        d = "0" if v == "sched-serial" else _ms(diff(v, "sched-serial"))
        lines.append(f"| {v} | {_ms(sl(v))} | {_ms(sl(v, 'E1_2'))} | "
                     f"{_ms(sl(v, 'R1_5'))} | {_ms(sl(v, 'A2_5'))} | {d} |")
    lines += ["", "## Check interval (only the used forward rounds executed)",
              "", "| forward checks | serial E1 | parallel E1 | "
              "parallel-nr E1 | serial R1 | parallel R1 | parallel-nr R1 | "
              "parallel-nr A2 |", "|---|---|---|---|---|---|---|---|"]
    labels = [("every round", "sched-{t}")]
    labels += [(f"unif={k}", f"int-{{t}}-unif{k}") for k in INTERVALS]
    labels += [(f"const={k}", f"int-{{t}}-const{k}") for k in CONST_KS]
    labels += [("terminal only", "int-{t}-term")]
    for label, pat in labels:
        cells = [_ms(sl(pat.format(t=t))) for t in
                 ("serial", "parallel", "parallel-nr")]
        cells += [_ms(sl(pat.format(t=t), "R1_5")) for t in
                  ("serial", "parallel", "parallel-nr")]
        cells.append(_ms(sl(pat.format(t="parallel-nr"), "A2_5")))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines.append(f"| none (const0) | {_ms(sl('const0'))} | | | "
                 f"{_ms(sl('const0', 'R1_5'))} | | | "
                 f"{_ms(sl('const0', 'A2_5'))} |")
    lines += ["", "## Inverse-half checks (forward unif=k + inverse every k)",
              "", "| schedule | k | E1 | R1 | Δ E1 vs forward-only same k |",
              "|---|---|---|---|---|"]
    for t in ("serial", "parallel"):
        for k in INVERSE_KS:
            base = f"sched-{t}" if k == 1 else f"int-{t}-unif{k}"
            v = f"inv-{t}-k{k}"
            lines.append(f"| {t} | {k} | {_ms(sl(v))} | {_ms(sl(v, 'R1_5'))}"
                         f" | {_ms(diff(v, base))} |")
    lines += ["", "## Leave-one-out ablations (every round checked)", "",
              "| term removed | serial E1 | serial ΔE1 | serial ΔR1 | "
              "parallel E1 | parallel ΔE1 | parallel ΔR1 |",
              "|---|---|---|---|---|---|---|"]
    for name in V.ABLATIONS:
        cells = []
        for t in ("serial", "parallel"):
            v = f"sched-{t}" if name == "full" else f"abl-{t}-{name}"
            de = diff(v, f"sched-{t}")
            dr = diff(v, f"sched-{t}", "R1_5")
            cells += [_ms(sl(v)), _ms(de), _ms(dr)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    if any(r["variant"].startswith("scan-") for r in rows):
        lines += ["", "## Check-noise scan: E1 with check-round noise x C "
                  "(gate noise fixed)", "",
                  "| C | " + " | ".join(
                      f"{t} {lab}" for t in ("serial", "parallel-nr")
                      for lab in ("unif1", "unif2", "unif4", "unif11",
                                  "term")) + " | const0 |",
                  "|---|" + "---|" * 11]
        for c in CHECK_SCALES:
            cells = [_ms(sl(f"scan-{t}-{lab}-c{c:g}")) for t in
                     ("serial", "parallel-nr") for lab in
                     ("unif1", "unif2", "unif4", "unif11", "term")]
            lines.append(f"| {c:g} | " + " | ".join(cells)
                         + f" | {_ms(sl('const0'))} |")
        lines += ["", "Paired ΔE1 = E1(checks) - E1(const0) at small C:", ""]
        for c in CHECK_SCALES[:4]:
            for t in ("serial", "parallel-nr"):
                lines.append(f"- C={c:g}, {t} unif1: "
                             f"{_ms(diff(f'scan-{t}-unif1-c{c:g}', 'const0'))}")
    (out / "first_order_tables.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'first_order_tables.md'}")
    return 0


GRID_P_INDEX = {5e-4: 3, 2e-3: 5, 5e-3: 6}
CROSSCHECK_PAIRS = (
    ("sched-serial", "mld5", "mld5"), ("sched-serial", "mld2", "mld2"),
    ("sched-serial", "lrbd", "lrbd_unif_1"),
    ("sched-serial", "lrbd_final", "lrbd_unif_1_final"),
    ("sched-serial", "lrbd_const_1", "lrbd_const_1"),
    ("sched-serial", "lrbd_unif_4", "lrbd_unif_4"),
    ("const0", "mld5", "const0_mld"), ("const0", "lrbd", "const0"))


def cmd_crosscheck(args) -> int:
    """Two-proportion chi-square of our counts against the lead's grid_main.

    ``sched-serial`` and ``const0`` build the same circuits as the lead's
    grid (tested), with different seeds, so per-(sequence, depth) error and
    acceptance fractions must agree within binomial noise.
    """
    from scipy import stats
    out = Path(args.out)
    grid_dir = Path(args.grid)
    lines = ["variant,method,grid_method,p,chi2,dof,p_value,max_abs_z"]
    for p, index in GRID_P_INDEX.items():
        grid = E.PointResult.from_npz(str(grid_dir / f"p{index}.npz"))
        for name, mine, theirs in CROSSCHECK_PAIRS:
            counts, n_seq = load_counts(out, name, p, args.num_seq)
            if n_seq < args.num_seq or theirs not in grid.tables:
                continue
            table = grid.tables[theirs]
            chi2, dof, worst = 0.0, 0, 0.0
            for s in range(n_seq):
                for d in range(len(grid.depths)):
                    for n1, k1, n2, k2 in (
                            (counts[f"{mine}__accepted"][s, d],
                             counts[f"{mine}__errors"][s, d],
                             table.accepted[s, d], table.errors[s, d]),
                            (counts[f"{mine}__shots"][s, d],
                             counts[f"{mine}__accepted"][s, d],
                             table.shots[s, d], table.accepted[s, d])):
                        if n1 == 0 or n2 == 0:
                            continue
                        pool = (k1 + k2) / (n1 + n2)
                        if pool <= 0.0 or pool >= 1.0:
                            continue
                        z = (k1 / n1 - k2 / n2) / np.sqrt(
                            pool * (1 - pool) * (1 / n1 + 1 / n2))
                        chi2 += z * z
                        dof += 1
                        worst = max(worst, abs(z))
            pval = stats.chi2.sf(chi2, dof) if dof else float("nan")
            lines.append(f"{name},{mine},{theirs},{p:g},{chi2:.1f},{dof},"
                         f"{pval:.3f},{worst:.2f}")
    (out / "crosscheck_grid_main.csv").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    cc = sub.add_parser("crosscheck")
    cc.add_argument("--out", required=True)
    cc.add_argument("--grid", required=True)
    cc.add_argument("--num-seq", type=int, default=10)
    fot = sub.add_parser("first-order-tables")
    fot.add_argument("--out", required=True)
    fo = sub.add_parser("first-order")
    fo.add_argument("--out", required=True)
    fo.add_argument("--groups",
                    default="schedules,intervals,inverse,ablations")
    fo.add_argument("--num-seq", type=int, default=10)
    fo.add_argument("--workers", type=int, default=4)
    fo.add_argument("--tag", default="",
                    help="suffix for the output file names")
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--groups", default="schedules,intervals,inverse,ablations")
    run.add_argument("--probs", default=",".join(map(str, PROBS)))
    run.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    run.add_argument("--num-seq", type=int, default=10)
    run.add_argument("--seq-start", type=int, default=0)
    run.add_argument("--shots", type=int, default=1_000_000)
    run.add_argument("--mid-shots", type=int, default=300_000)
    run.add_argument("--decode-shots", type=int, default=100_000)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-minutes", type=float, default=15.0)
    run.add_argument("--only", default="",
                     help="comma-separated substrings of variant names")
    ana = sub.add_parser("analyze")
    ana.add_argument("--out", required=True)
    ana.add_argument("--groups", default="schedules,intervals,inverse,ablations")
    ana.add_argument("--probs", default=",".join(map(str, PROBS)))
    ana.add_argument("--num-seq", type=int, default=30)
    ana.add_argument("--num-boot", type=int, default=1000)
    ana.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    commands = {"run": cmd_run, "analyze": cmd_analyze,
                "first-order": cmd_first_order,
                "first-order-tables": cmd_first_order_tables,
                "crosscheck": cmd_crosscheck}
    return commands[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
