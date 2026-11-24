"""Work package K: Steane [[7,1,3]] LRB with correction vs detection.

Subcommands (run from the repository root after ``module load python/3.11``):

``checks``
    Circuit-level distance of the four circuit families (naive / flag
    extraction x sequential / parallel measurement) with an exact pair check
    of the DEM, Stim's ``search_for_undetectable_logical_errors`` and
    ``shortest_graphlike_error``; exact first-order DEM coefficients (U1, R1,
    E1, A2 as in notes/L0 section 4) over sequences and depths at p = 1e-6.
``run``
    Sample (extraction, schedule, p, sequence) tasks in a process pool. Each
    task writes ``<out>/parts/<ext>_<schedule>_p<p>_s<seq>.npz`` (an
    ``experiment.PointResult`` for one sequence); finished parts are skipped.
``analyze``
    Merge parts, fit ``F = b f^m`` (the repo's offset-0 fit) with a sequence
    bootstrap, compare with exact physical RB at p_RB = p and 5p and with the
    d=2 [[5,1,2]] main grid, find pseudo-thresholds, and write
    ``summary.json``, ``fits.csv``, ``curves.csv`` and ``tables.md``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

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

from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import fits as FT  # noqa: E402
from lrb_qubit import steane_circuits as SV  # noqa: E402
from lrb_qubit import steane_code as SC  # noqa: E402

DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
P_VALUES = [1e-4, 5e-4, 1e-3, 2e-3, 5e-3]
SEQ_SEED = 20261028
VARIANTS = [("naive", "sequential"), ("naive", "parallel"),
            ("flag", "sequential"), ("flag", "parallel")]
GRID_MAIN = ROOT / "studies/correction-decoders/qubit/results/grid_main"
LLR_THRESHOLDS = [2.0, 4.0, 6.0, 8.0]
REPORT_METHODS = ["mld", "lrbd_unif_1_final", "lrbd_unif_1",
                  "lrbd_const_1_final", "lrbd_unif_1_final_noflag",
                  "const0", "const0_mld"]


def part_path(out: Path, ext: str, schedule: str, p: float, seq: int) -> Path:
    return out / "parts" / f"{ext}_{schedule}_p{p:g}_s{seq:02d}.npz"


# ----------------------------------------------------------------------
# run
# ----------------------------------------------------------------------
def _task(args):
    (ext, schedule, p, seq_index, sequence, depths, shots, decode_shots,
     const0, seed, path) = args
    start = time.perf_counter()
    code = SC.SteaneCode(ext)
    result = SV.run_sequence(code, schedule, p, sequence, depths, shots,
                             decode_shots=decode_shots, const0=const0,
                             seed=seed)
    tmp = str(path) + ".tmp.npz"
    result.to_npz(tmp)
    os.replace(tmp, path)
    return str(path), time.perf_counter() - start


def cmd_run(args) -> None:
    out = Path(args.out)
    (out / "parts").mkdir(parents=True, exist_ok=True)
    sequences = E.clifford_sequences(args.num_seq, max(DEPTHS), SEQ_SEED)
    p_values = [float(x) for x in args.p_values.split(",")]
    variants = [tuple(v.split(":")) for v in args.variants.split(",")]
    meta_path = out / "meta.json"
    meta = {"depths": DEPTHS, "p_values": p_values, "num_seq": args.num_seq,
            "seq_seed": SEQ_SEED, "variants": [list(v) for v in variants],
            "sequences": sequences, "codes": {
                ext: SC.SteaneCode(ext).describe()
                for ext in ("naive", "flag")}, "runs": []}
    if meta_path.exists():
        old = json.loads(meta_path.read_text())
        meta["p_values"] = sorted(set(old["p_values"]) | set(p_values))
        meta["variants"] = old["variants"] + [
            list(v) for v in variants if list(v) not in old["variants"]]
        meta["runs"] = old.get("runs", [])
        if old["num_seq"] > meta["num_seq"]:
            meta["num_seq"] = old["num_seq"]
            meta["sequences"] = old["sequences"]
    meta["runs"].append({"p_values": p_values,
                         "variants": [list(v) for v in variants],
                         "num_seq": args.num_seq, "shots": args.shots,
                         "decode_shots": args.decode_shots, "seed": args.seed})
    meta_path.write_text(json.dumps(meta, indent=1))
    tasks = []
    for ext, schedule in variants:
        for p_index, p in enumerate(p_values):
            for seq in range(args.num_seq):
                path = part_path(out, ext, schedule, p, seq)
                if path.exists():
                    continue
                # const0 does not depend on the extraction or the schedule.
                const0 = (ext, schedule) == variants[0]
                seed = (args.seed + 7919 * seq + 104729 * p_index
                        + 15485863 * VARIANTS.index((ext, schedule)))
                tasks.append((ext, schedule, p, seq, sequences[seq], DEPTHS,
                              args.shots, args.decode_shots, const0, seed,
                              path))
    # Slowest first: flagged decoding at large p builds many flag patterns.
    tasks.sort(key=lambda t: (t[0] != "flag", -t[2]))
    print(f"{len(tasks)} tasks", flush=True)
    deadline = time.time() + 60 * args.max_minutes
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        queue = list(tasks)
        while queue and len(pending) < args.workers:
            task = queue.pop(0)
            pending[pool.submit(_task, task)] = task
        while pending:
            done = next(as_completed(pending))
            pending.pop(done)
            path, seconds = done.result()
            print(f"{Path(path).name} {seconds:.1f}s", flush=True)
            if queue and time.time() < deadline:
                task = queue.pop(0)
                pending[pool.submit(_task, task)] = task
    if queue:
        print(f"deadline reached; {len(queue)} tasks left", flush=True)


# ----------------------------------------------------------------------
# checks
# ----------------------------------------------------------------------
def _first_order_task(args):
    ext, schedule, seq_index, sequence, depths, p = args
    code = SC.SteaneCode(ext)
    rows = []
    for depth in depths:
        lc = SV.build_lrb_circuit(code, sequence, depth, p, schedule=schedule)
        summary = SV.first_order_summary(lc.circuit.detector_error_model())
        rows.append({"extraction": ext, "schedule": schedule,
                     "seq": seq_index, "depth": depth,
                     **{k: summary[k] / p for k in ("U1", "R1", "E1")},
                     "A2": summary["A2"] / p ** 2,
                     "ambiguous_classes": summary["ambiguous_classes"]})
    return rows


def cmd_checks(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sequences = E.clifford_sequences(args.num_seq, max(DEPTHS), SEQ_SEED)
    distance = []
    for ext, schedule in VARIANTS:
        code = SC.SteaneCode(ext)
        for depth in (0, 1, 2, 4):
            lc = SV.build_lrb_circuit(code, sequences[0], depth, 1e-3,
                                      schedule=schedule)
            start = time.perf_counter()
            report = SV.distance_report(lc.circuit, search_size=6,
                                        search_degree=6)
            report.update(extraction=ext, schedule=schedule, depth=depth,
                          seconds=time.perf_counter() - start)
            distance.append(report)
            print(ext, schedule, depth, report["exact_lower_bound"],
                  report.get("stim_search"), report["shortest_graphlike"],
                  flush=True)
    (out / "distance.json").write_text(json.dumps(distance, indent=1))

    p = 1e-6
    jobs = [(ext, schedule, s, sequences[s], DEPTHS, p)
            for ext, schedule in VARIANTS for s in range(args.num_seq)]
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_first_order_task, j)
                                 for j in jobs]):
            rows.extend(fut.result())
    rows.sort(key=lambda r: (r["extraction"], r["schedule"], r["seq"],
                             r["depth"]))
    slopes = {}
    for ext, schedule in VARIANTS:
        sub = [r for r in rows if r["extraction"] == ext
               and r["schedule"] == schedule]
        entry = {}
        for key in ("R1", "E1", "A2", "U1"):
            per_seq = []
            for s in range(args.num_seq):
                pts = sorted((r["depth"], r[key]) for r in sub if r["seq"] == s)
                x = np.array([a for a, _ in pts], float)
                y = np.array([b for _, b in pts], float)
                per_seq.append(np.polyfit(x, y, 1)[0])
            per_seq = np.array(per_seq)
            depth0 = np.mean([r[key] for r in sub if r["depth"] == 0])
            depth22 = np.mean([r[key] for r in sub if r["depth"] == 22])
            entry[key] = {"slope_mean": float(per_seq.mean()),
                          "slope_sem": float(per_seq.std(ddof=1)
                                             / np.sqrt(len(per_seq))),
                          "depth0_mean": float(depth0),
                          "depth22_mean": float(depth22)}
        slopes[f"{ext}:{schedule}"] = entry
    (out / "first_order.json").write_text(json.dumps(
        {"p": p, "units": "R1, E1, U1 in units of p; A2 in units of p^2",
         "slopes_per_step": slopes, "rows": rows}, indent=1))
    print(json.dumps(slopes, indent=1))


# ----------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------
def _merge(out: Path, ext: str, schedule: str, p: float, num_seq: int
           ) -> E.PointResult | None:
    merged = None
    for seq in range(num_seq):
        path = part_path(out, ext, schedule, p, seq)
        if not path.exists():
            return None
        part = E.PointResult.from_npz(str(path))
        if merged is None:
            merged = E.PointResult(p, part.depths, num_seq)
        for name, table in part.tables.items():
            target = merged.table(name)
            target.shots[seq] = table.shots[0]
            target.accepted[seq] = table.accepted[0]
            target.errors[seq] = table.errors[0]
        for name, hist in part.gap_hist.items():
            merged.hist(name)[seq] = hist[0]
    return merged


def _merge_const0(out: Path, variants, p: float, num_seq: int):
    for ext, schedule in variants:
        merged = _merge(out, ext, schedule, p, num_seq)
        if merged is not None and "const0" in merged.tables:
            return merged
    return None


def threshold_counts(hist: np.ndarray, tau: float) -> tuple[np.ndarray,
                                                            np.ndarray]:
    """Accepted and wrong counts for ``|LLR| >= tau`` from gap histograms."""
    start = int(np.searchsorted(E.GAP_EDGES, tau - 1e-12))
    sel = hist[:, :, start:, :]
    accepted = sel.sum(axis=(2, 3))
    wrong = sel[..., 1].sum(axis=2)
    return accepted, wrong


def bootstrap_eps(depths, accepted, errors, num_boot, seed=0, picks=None):
    """Error per step ``(1 - f)/2`` with a sequence bootstrap."""
    point = FT.fit_from_counts(depths, accepted, errors, "lrb")
    n = accepted.shape[0]
    rng = np.random.default_rng(seed)
    values = []
    if picks is None:
        picks = [rng.integers(0, n, size=n) for _ in range(num_boot)]
    for pick in picks:
        fit = FT.fit_from_counts(depths, accepted[pick], errors[pick], "lrb")
        values.append(np.nan if fit is None else (1 - fit["f"]) / 2)
    values = np.array(values)
    eps = np.nan if point is None else (1 - point["f"]) / 2
    finite = values[np.isfinite(values)]
    return {"eps": float(eps),
            "eps_std": float(np.std(finite)) if len(finite) > 1 else np.nan,
            "eps_lo": float(np.percentile(finite, 16)) if len(finite) else np.nan,
            "eps_hi": float(np.percentile(finite, 84)) if len(finite) else np.nan,
            "n_points": 0 if point is None else int(point["n_points"]),
            "boot": values}


def acceptance_per_step(depths, shots, accepted):
    """Least-squares slope of log(mean acceptance) vs depth (as exp)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        acc = np.mean(accepted / np.maximum(shots, 1), axis=0)
    ok = acc > 0
    if ok.sum() < 2:
        return np.nan
    slope = np.polyfit(np.asarray(depths, float)[ok], np.log(acc[ok]), 1)[0]
    return float(np.exp(slope))


def fit_exact_fidelity(depths, fid, num_boot, seed=0, picks=None):
    """Offset-0 fit of exact per-sequence fidelities with a bootstrap."""
    def one(f):
        mean, std = np.mean(f, axis=0), np.std(f, axis=0)
        fit = FT.fit_fixed_offset(depths, mean, std)
        return np.nan if fit is None else (1 - fit["f"]) / 2
    n = fid.shape[0]
    rng = np.random.default_rng(seed)
    if picks is None:
        picks = [rng.integers(0, n, size=n) for _ in range(num_boot)]
    values = np.array([one(fid[pick]) for pick in picks])
    return {"eps": float(one(fid)), "eps_std": float(np.nanstd(values)),
            "boot": values}


def crossing(p_values, ratio):
    """Smallest p where log(ratio) crosses zero going up (log-log interp)."""
    p = np.asarray(p_values, float)
    r = np.asarray(ratio, float)
    ok = np.isfinite(r) & (r > 0)
    p, r = p[ok], r[ok]
    if len(p) < 2:
        return np.nan, "insufficient"
    g = np.log(r)
    if g[0] >= 0:
        return float(p[0]), "below_grid"
    for i in range(len(p) - 1):
        if g[i] < 0 <= g[i + 1]:
            x0, x1 = np.log(p[i]), np.log(p[i + 1])
            x = x0 + (x1 - x0) * (-g[i]) / (g[i + 1] - g[i])
            return float(np.exp(x)), "interpolated"
    return float(p[-1]), "above_grid"


def extrapolated_crossing(p_values, eps, eps_phys):
    """Crossing below the grid from a power law through the two lowest p.

    ``eps = a p^k`` through the two lowest points, ``eps_phys = c p``;
    returns ``(c / a)^(1 / (k - 1))`` (needs ``k > 1``) and ``k``.
    """
    p = np.asarray(p_values[:2], float)
    e = np.asarray(eps[:2], float)
    c = float(eps_phys[0]) / p[0]
    if not (np.all(e > 0) and np.all(np.isfinite(e))):
        return np.nan, np.nan
    k = np.log(e[1] / e[0]) / np.log(p[1] / p[0])
    a = e[0] / p[0] ** k
    if k <= 1:
        return np.nan, float(k)
    return float((c / a) ** (1.0 / (k - 1.0))), float(k)


def cmd_analyze(args) -> None:
    out = Path(args.out)
    meta = json.loads((out / "meta.json").read_text())
    depths = meta["depths"]
    p_values = [float(x) for x in args.p_values.split(",")]
    num_seq = args.num_seq
    sequences = meta["sequences"][:num_seq]
    rng = np.random.default_rng(args.seed)
    picks = [rng.integers(0, num_seq, size=num_seq)
             for _ in range(args.num_boot)]
    variants = [tuple(v) for v in meta["variants"]]
    if args.variants:
        variants = [tuple(v.split(":")) for v in args.variants.split(",")]
    tag = f"_{args.tag}" if args.tag else ""

    fits_rows, curve_rows = [], []
    summary = {"depths": depths, "p_values": p_values, "num_seq": num_seq,
               "num_boot": args.num_boot, "methods": {}}
    boot_store = {}

    def record(key, p, entry, extra):
        row = {"method": key, "p": p, "eps": entry["eps"],
               "eps_std": entry["eps_std"],
               "eps_over_p": entry["eps"] / p,
               "eps_std_over_p": entry["eps_std"] / p, **extra}
        fits_rows.append(row)
        summary["methods"].setdefault(key, {})[f"{p:g}"] = {
            k: v for k, v in row.items() if k not in ("method", "p")}
        boot_store[(key, p)] = entry["boot"]

    for p in p_values:
        # Physical RB, exact, same sequences.
        for scale, label in ((1.0, "phys_rb_p"), (5.0, "phys_rb_5p")):
            fid = np.array([[SV.physical_rb_fidelity(seq, m, scale * p)
                             for m in depths] for seq in sequences])
            entry = fit_exact_fidelity(depths, fid, args.num_boot,
                                       picks=picks)
            ens = (1 - SV.physical_rb_decay(scale * p)) / 2
            record(label, p, entry, {"ensemble_eps": ens,
                                     "acceptance_per_step": 1.0})
        const0_point = _merge_const0(out, variants, p, num_seq)
        for ext, schedule in variants:
            point = _merge(out, ext, schedule, p, num_seq)
            if point is None:
                print(f"missing parts for {ext} {schedule} p={p:g}")
                continue
            tables = dict(point.tables)
            if const0_point is not None:
                for name in ("const0", "const0_mld"):
                    if name in const0_point.tables:
                        tables[name] = const0_point.tables[name]
            hist = point.gap_hist.get("mld")
            methods = [m for m in REPORT_METHODS if m in tables]
            for method in methods:
                t = tables[method]
                entry = bootstrap_eps(depths, t.accepted, t.errors,
                                      args.num_boot, picks=picks)
                acc = acceptance_per_step(depths, t.shots, t.accepted)
                key = f"{ext}:{schedule}:{method}"
                if method.startswith("const0"):
                    key = f"steane:{method}"
                    if any(r["method"] == key and r["p"] == p
                           for r in fits_rows):
                        continue
                record(key, p, entry, {"acceptance_per_step": acc})
                fid = FT.per_sequence_fidelity(t.accepted, t.errors)
                with np.errstate(invalid="ignore", divide="ignore"):
                    rej = 1 - t.accepted / np.maximum(t.shots, 1)
                for d_i, m in enumerate(depths):
                    curve_rows.append({
                        "method": key, "p": p, "depth": m,
                        "F_mean": float(np.nanmean(fid[:, d_i]))
                        if np.isfinite(fid[:, d_i]).any() else np.nan,
                        "F_std": float(np.nanstd(fid[:, d_i]))
                        if np.isfinite(fid[:, d_i]).any() else np.nan,
                        "acceptance": float(np.mean(1 - rej[:, d_i])),
                        "shots": int(t.shots[:, d_i].sum()),
                        "accepted": int(t.accepted[:, d_i].sum()),
                        "errors": int(t.errors[:, d_i].sum())})
            if hist is not None:
                for tau in LLR_THRESHOLDS:
                    acc_c, err_c = threshold_counts(hist, tau)
                    shots = tables["mld"].shots
                    entry = bootstrap_eps(depths, acc_c, err_c,
                                          args.num_boot, picks=picks)
                    acc = acceptance_per_step(depths, shots, acc_c)
                    record(f"{ext}:{schedule}:mld_llr{tau:g}", p, entry,
                           {"acceptance_per_step": acc})

    # d = 2 [[5,1,2]] main grid at the same p (first num_seq sequences and
    # all 30), same fitter.
    d2 = {}
    if GRID_MAIN.exists():
        gmeta = json.loads((GRID_MAIN / "meta.json").read_text())
        for p in p_values:
            if p not in gmeta["probs"]:
                continue
            idx = gmeta["probs"].index(p)
            path = GRID_MAIN / f"p{idx}.npz"
            if not path.exists():
                continue
            point = E.PointResult.from_npz(str(path))
            for method in ("mld5", "lrbd_unif_1_final", "lrbd_unif_1",
                           "lrbd_const_1", "const0_mld"):
                if method not in point.tables:
                    continue
                t = point.tables[method]
                for n_use, subset_tag in ((num_seq, f"first{num_seq}"),
                                   (t.accepted.shape[0], "all")):
                    sub_rng = np.random.default_rng(args.seed + 1)
                    sub_picks = [sub_rng.integers(0, n_use, size=n_use)
                                 for _ in range(args.num_boot)]
                    entry = bootstrap_eps(depths, t.accepted[:n_use],
                                          t.errors[:n_use], args.num_boot,
                                          picks=sub_picks)
                    acc = acceptance_per_step(depths, t.shots[:n_use],
                                              t.accepted[:n_use])
                    d2.setdefault(f"d2:{method}:{subset_tag}", {})[f"{p:g}"] = {
                        "eps": entry["eps"], "eps_std": entry["eps_std"],
                        "eps_over_p": entry["eps"] / p,
                        "eps_std_over_p": entry["eps_std"] / p,
                        "acceptance_per_step": acc,
                        "num_seq": int(n_use)}
    summary["d2_main_grid"] = d2

    # Pseudo-thresholds: corrected error per step against physical RB.
    thresholds = {}
    for ext, schedule in variants:
        key = f"{ext}:{schedule}:mld"
        if not all((key, p) in boot_store for p in p_values):
            continue
        for label in ("phys_rb_p", "phys_rb_5p"):
            ratio = [summary["methods"][key][f"{p:g}"]["eps"]
                     / summary["methods"][label][f"{p:g}"]["eps"]
                     for p in p_values]
            p_star, status = crossing(p_values, ratio)
            boot = []
            for b in range(args.num_boot):
                r_b = [boot_store[(key, p)][b] / boot_store[(label, p)][b]
                       for p in p_values]
                ps, st = crossing(p_values, r_b)
                if st == "interpolated":
                    boot.append(ps)
            boot = np.array(boot)
            extra = {}
            if status == "below_grid":
                eps_l = [summary["methods"][key][f"{p:g}"]["eps"]
                         for p in p_values]
                eps_p = [summary["methods"][label][f"{p:g}"]["eps"]
                         for p in p_values]
                p_ext, k_ext = extrapolated_crossing(p_values, eps_l, eps_p)
                ext_boot = []
                for b in range(args.num_boot):
                    eb = [boot_store[(key, p)][b] for p in p_values]
                    pb = [boot_store[(label, p)][b] for p in p_values]
                    val, _ = extrapolated_crossing(p_values, eb, pb)
                    if np.isfinite(val):
                        ext_boot.append(val)
                extra = {"p_star_extrapolated": p_ext,
                         "low_p_exponent_two_point": k_ext,
                         "extrapolated_lo": float(np.percentile(ext_boot, 16))
                         if ext_boot else np.nan,
                         "extrapolated_hi": float(np.percentile(ext_boot, 84))
                         if ext_boot else np.nan}
            thresholds[f"{key} vs {label}"] = {
                **extra,
                "p_star": p_star, "status": status,
                "ratio_at_grid": dict(zip([f"{p:g}" for p in p_values],
                                          map(float, ratio))),
                "boot_median": float(np.median(boot)) if len(boot) else np.nan,
                "boot_lo": float(np.percentile(boot, 16)) if len(boot) else np.nan,
                "boot_hi": float(np.percentile(boot, 84)) if len(boot) else np.nan,
                "boot_fraction_interpolated": len(boot) / args.num_boot}
    summary["pseudo_thresholds"] = thresholds

    # Model crossing: eps(p) = e1 p + c2 p^2 with e1 the exact first-order
    # MLD coefficient per step (first_order.json; 0 for flags) and c2 fitted
    # on the grid points with p <= 1e-3 (inverse-variance weights).
    first_order_path = out / "first_order.json"
    model = {}
    if first_order_path.exists():
        slopes = json.loads(first_order_path.read_text())["slopes_per_step"]
        low = [p for p in p_values if p <= 1e-3]
        for ext, schedule in variants:
            key = f"{ext}:{schedule}:mld"
            e1 = slopes.get(f"{ext}:{schedule}", {}).get("E1", {}).get(
                "slope_mean", np.nan)
            if key not in summary["methods"] or not np.isfinite(e1):
                continue

            def fit_c2(eps_by_p, std_by_p):
                c, w = [], []
                for p in low:
                    e, s = eps_by_p[p], std_by_p[p]
                    if not (np.isfinite(e) and np.isfinite(s) and s > 0):
                        continue
                    c.append((e - e1 * p) / p ** 2)
                    w.append((p ** 2 / s) ** 2)
                if not c:
                    return np.nan
                return float(np.average(c, weights=w))

            eps = {p: summary["methods"][key][f"{p:g}"]["eps"] for p in low}
            std = {p: summary["methods"][key][f"{p:g}"]["eps_std"] for p in low}
            c2 = fit_c2(eps, std)
            c2_boot = np.array([fit_c2({p: boot_store[(key, p)][b] for p in low},
                                       std) for b in range(args.num_boot)])
            entry = {"e1": e1, "c2": c2,
                     "c2_std": float(np.nanstd(c2_boot)),
                     "c2_per_point": {f"{p:g}": (eps[p] - e1 * p) / p ** 2
                                      for p in low}}
            for label in ("phys_rb_p", "phys_rb_5p"):
                c_phys = np.mean([summary["methods"][label][f"{p:g}"]["eps"] / p
                                  for p in p_values])
                if c_phys <= e1:
                    entry[label] = {"c_phys": float(c_phys), "p_star": None,
                                    "note": "first-order term alone exceeds "
                                            "physical RB: no crossing"}
                    continue
                stars = (c_phys - e1) / c2_boot[np.isfinite(c2_boot)
                                                & (c2_boot > 0)]
                entry[label] = {
                    "c_phys": float(c_phys),
                    "p_star": (float((c_phys - e1) / c2)
                               if np.isfinite(c2) and c2 > 0 else None),
                    "p_star_lo": (float(np.percentile(stars, 16))
                                  if len(stars) else np.nan),
                    "p_star_hi": (float(np.percentile(stars, 84))
                                  if len(stars) else np.nan)}
            model[key] = entry
    summary["model_crossings"] = model

    # Power-law exponent of the corrected error per step at the lowest p.
    exponents = {}
    for ext, schedule in variants:
        key = f"{ext}:{schedule}:mld"
        if key not in summary["methods"]:
            continue
        pts = [(p, summary["methods"][key][f"{p:g}"]["eps"]) for p in p_values
               if f"{p:g}" in summary["methods"][key]]
        low = [(p, e) for p, e in pts if p <= 1e-3 and e > 0]
        if len(low) >= 2:
            x = np.log([p for p, _ in low])
            y = np.log([e for _, e in low])
            exponents[key] = float(np.polyfit(x, y, 1)[0])
    summary["low_p_exponent"] = exponents

    import csv
    with open(out / f"fits{tag}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(
            {k for r in fits_rows for k in r}))
        writer.writeheader()
        writer.writerows(fits_rows)
    with open(out / f"curves{tag}.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(curve_rows[0]))
        writer.writeheader()
        writer.writerows(curve_rows)
    (out / f"summary{tag}.json").write_text(json.dumps(summary, indent=1,
                                                       default=float))
    write_tables(out, summary, p_values, tag)
    print((out / f"tables{tag}.md").read_text())


def _fmt(entry, scale=1.0, digits=3):
    if entry is None or not np.isfinite(entry.get("eps", np.nan)):
        return "–"
    v, s = entry["eps_over_p"] * scale, entry["eps_std_over_p"] * scale
    if not np.isfinite(s):
        return f"{v:.{digits}g}"
    return f"{v:.{digits}g} ± {s:.2g}"


def write_tables(out: Path, summary: dict, p_values, tag: str = "") -> None:
    lines = ["# WP K tables (generated by steane_compare.py analyze)", ""]
    header = "| method | " + " | ".join(f"p={p:g}" for p in p_values) + " |"
    sep = "|---|" + "---|" * len(p_values)
    lines += ["## Error per depth step (1 - f)/2, in units of p "
              "(sequence-bootstrap std)", "", header, sep]
    methods = summary["methods"]
    order = [k for k in methods if k.startswith("phys")] + sorted(
        k for k in methods if not k.startswith("phys"))
    for key in order:
        cells = [_fmt(methods[key].get(f"{p:g}")) for p in p_values]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    for key in sorted(summary["d2_main_grid"]):
        cells = [_fmt(summary["d2_main_grid"][key].get(f"{p:g}"))
                 for p in p_values]
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "## Acceptance per depth step (log-linear fit of the mean "
              "acceptance)", "", header, sep]
    for key in order:
        if key.startswith("phys"):
            continue
        cells = []
        for p in p_values:
            e = methods[key].get(f"{p:g}")
            a = np.nan if e is None else e.get("acceptance_per_step", np.nan)
            cells.append("–" if not np.isfinite(a) else f"{a:.4f}")
        lines.append(f"| {key} | " + " | ".join(cells) + " |")
    lines += ["", "## Pseudo-thresholds (corrected error per step = physical "
              "RB error per step)", "",
              "| comparison | p* (point) | status | bootstrap median "
              "[16%, 84%] | fraction of replicates with a crossing |",
              "|---|---|---|---|---|"]
    for key, t in summary["pseudo_thresholds"].items():
        status = t["status"]
        if "p_star_extrapolated" in t:
            status += (f"; power-law extrapolation {t['p_star_extrapolated']:.3g}"
                       f" [{t['extrapolated_lo']:.3g}, {t['extrapolated_hi']:.3g}]"
                       f", exponent {t['low_p_exponent_two_point']:.2f}")
        lines.append(
            f"| {key} | {t['p_star']:.3g} | {status} | "
            f"{t['boot_median']:.3g} [{t['boot_lo']:.3g}, "
            f"{t['boot_hi']:.3g}] | {t['boot_fraction_interpolated']:.2f} |")
    lines += ["", "## Model crossings: eps = e1 p + c2 p^2 (e1 exact, c2 "
              "fitted at p <= 1e-3)", "",
              "| variant | e1 (/p) | c2 (/p^2) | p* vs RB at p | "
              "p* vs RB at 5p |", "|---|---|---|---|---|"]
    for key, m in summary.get("model_crossings", {}).items():
        cells = []
        for label in ("phys_rb_p", "phys_rb_5p"):
            e = m.get(label, {})
            if e.get("p_star") is None:
                cells.append("none (e1 > physical)" if "note" in e
                             else "no fit")
            else:
                cells.append(f"{e['p_star']:.3g} [{e['p_star_lo']:.3g}, "
                             f"{e['p_star_hi']:.3g}]")
        lines.append(f"| {key} | {m['e1']:.3f} | {m['c2']:.4g} ± "
                     f"{m['c2_std']:.2g} | " + " | ".join(cells) + " |")
    lines += ["", "## Low-p exponent of the corrected error per step "
              "(log-log slope over p <= 1e-3)", ""]
    for key, k in summary["low_p_exponent"].items():
        lines.append(f"- {key}: {k:.2f}")
    (out / f"tables{tag}.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--out", required=True)
    run.add_argument("--p-values", default=",".join(f"{p:g}" for p in P_VALUES))
    run.add_argument("--variants", default=",".join(
        f"{e}:{s}" for e, s in VARIANTS))
    run.add_argument("--num-seq", type=int, default=10)
    run.add_argument("--shots", type=int, default=200_000)
    run.add_argument("--decode-shots", type=int, default=100_000)
    run.add_argument("--workers", type=int, default=4)
    run.add_argument("--max-minutes", type=float, default=17.0)
    run.add_argument("--seed", type=int, default=1)
    chk = sub.add_parser("checks")
    chk.add_argument("--out", required=True)
    chk.add_argument("--num-seq", type=int, default=10)
    chk.add_argument("--workers", type=int, default=4)
    ana = sub.add_parser("analyze")
    ana.add_argument("--out", required=True)
    ana.add_argument("--p-values", default=",".join(f"{p:g}" for p in P_VALUES))
    ana.add_argument("--num-seq", type=int, default=10)
    ana.add_argument("--num-boot", type=int, default=200)
    ana.add_argument("--seed", type=int, default=0)
    ana.add_argument("--variants", default="",
                     help="comma list ext:schedule (default: all in meta)")
    ana.add_argument("--tag", default="", help="suffix for output files")
    args = parser.parse_args()
    {"run": cmd_run, "checks": cmd_checks, "analyze": cmd_analyze}[args.cmd](
        args)


if __name__ == "__main__":
    main()
