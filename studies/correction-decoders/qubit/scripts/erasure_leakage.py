"""WP I driver: erasure heralding, leakage and flags for qubit [[5,1,2]] LRB.

Subcommands (all write under ``--out``, default
``studies/correction-decoders/qubit/results/I``):

``erasure``
    ``erasure.run_erasure_point`` over p x erasure configurations, one task
    per (configuration, p, sequence); every method is scored on the same
    shots. Parts go to ``erasure/parts/<config>_p<i>_s<j>.npz``.
``leakage``
    ``leakage.run_leakage_point`` over p x leakage configurations. Parts go
    to ``leakage/parts/``.
``flags``
    Sampled exact-MLD logical error with and without flag qubits (each
    flagged circuit decoded by the lead's ``TransferMatrixMLD`` with flag
    detectors in their round's slot). Parts go to ``flags/parts/``.
``firstorder``
    Exact single-fault analysis: which single faults the exact MLD
    miscorrects, by fault kind, with and without flags; first-order
    coefficient of the herald-aware MLD versus the erasure fraction; check
    that every single heralded erasure is corrected. Writes
    ``firstorder/*.json``.
``report``
    Merge the parts and write ``tables.md`` plus CSVs.

Finished parts are skipped, so an interrupted run resumes. Example (shared
allocation, 24 cores)::

    srun --jobid=<ID> --overlap --cpu-bind=none -N 1 -n 1 -c 256 bash -c \
      "module load python/3.11; cd <WORKTREE>; \
      export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1; \
      python studies/correction-decoders/qubit/scripts/erasure_leakage.py erasure --workers 24"

(``--cpu-bind=none -c 256``: without it, overlapping steps of different
users of the shared node are pinned to the same few CPUs.)
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

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import erasure as ER  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import leakage as L  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

DEFAULT_OUT = ROOT / "studies/correction-decoders/qubit/results/I"
PROBS = [5e-4, 2e-3, 5e-3]
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
SEQ_SEED = 20261028

ERASURE_CONFIGS = {
    "r0": ER.ErasureSpec(),
    "2q_r0.5": ER.ErasureSpec(two_qubit=0.5),
    "2q_r0.9": ER.ErasureSpec(two_qubit=0.9),
    "2q_r1": ER.ErasureSpec(two_qubit=1.0),
    "all_r0.5": ER.ErasureSpec(two_qubit=0.5, meas_idle=0.5),
    "all_r0.9": ER.ErasureSpec(two_qubit=0.9, meas_idle=0.9),
    "all_r1": ER.ErasureSpec(two_qubit=1.0, meas_idle=1.0),
}
LEAKAGE_CONFIGS = {
    "none": L.LeakageModel(rate=0.0),
    "k0.1": L.LeakageModel(rate=0.1),
    "k0.1_lru": L.LeakageModel(rate=0.1, lru=True),
    "k0.5": L.LeakageModel(rate=0.5),
    "k0.5_lru": L.LeakageModel(rate=0.5, lru=True),
}
FLAG_CONFIGS = {
    "none": (None, "standard"),
    "ideal_standard": ("ideal", "standard"),
    "ideal_wrap": ("ideal", "wrap"),
    "noisy_standard": ("noisy", "standard"),
    "noisy_wrap": ("noisy", "wrap"),
}


def _seed(*parts: int) -> int:
    value = 7
    for part in parts:
        value = value * 1_000_003 + int(part)
    return value % (2 ** 31)


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


def _erasure_task(args):
    config, ci, pi, si, sequence, depths, shots, path = args
    start = time.perf_counter()
    result = ER.run_erasure_point(PROBS[pi], ERASURE_CONFIGS[config],
                                  [sequence], depths, shots,
                                  seed=_seed(1, ci, pi, si))
    arrays = {"p": PROBS[pi], "depths": np.array(depths)}
    for name, table in result["tables"].items():
        arrays[f"{name}__shots"] = table.shots[0]
        arrays[f"{name}__accepted"] = table.accepted[0]
        arrays[f"{name}__errors"] = table.errors[0]
    for name, hist in result["hists"].items():
        arrays[f"gap__{name}"] = hist[0]
    arrays["gap_edges"] = result["gap_edges"]
    arrays["herald_stats"] = result["herald_stats"][0]
    arrays["ambiguous"] = result["ambiguous"][0]
    np.savez_compressed(path, **arrays)
    return config, pi, si, time.perf_counter() - start


def _leakage_task(args):
    config, ci, pi, si, sequence, depths, shots, path = args
    start = time.perf_counter()
    result = L.run_leakage_point(PROBS[pi], LEAKAGE_CONFIGS[config],
                                 [sequence], depths, shots,
                                 seed=_seed(2, ci, pi, si))
    arrays = {"p": PROBS[pi], "depths": np.array(depths)}
    for name, table in result["tables"].items():
        arrays[f"{name}__shots"] = table.shots[0]
        arrays[f"{name}__accepted"] = table.accepted[0]
        arrays[f"{name}__errors"] = table.errors[0]
    arrays["leaked_trajectory"] = result["leaked_trajectory"]
    arrays["autocorr"] = result["autocorr"]
    arrays["flagged"] = result["flagged"][0]
    arrays["leak_events"] = result["leak_events"][0]
    np.savez_compressed(path, **arrays)
    return config, pi, si, time.perf_counter() - start


def _flag_task(args):
    config, ci, pi, si, sequence, depths, shots, path = args
    start = time.perf_counter()
    flags, layout = FLAG_CONFIGS[config]
    p = PROBS[pi]
    errors = np.zeros(len(depths), dtype=np.int64)
    counts = np.zeros(len(depths), dtype=np.int64)
    for di, depth in enumerate(depths):
        lc = ER.build_instrumented_lrb(sequence, depth, p, flags=flags,
                                       flag_layout=layout, mode="decode")
        decoder = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                      lc.decoder_slots())
        sampler = lc.circuit.compile_detector_sampler(
            seed=_seed(3, ci, pi, si, di))
        done = 0
        while done < shots:
            n = min(10_000, shots - done)
            dets, obs = sampler.sample(n, separate_observables=True)
            llr = decoder.log_likelihood_ratio(dets)
            errors[di] += int(((llr < 0) != obs[:, 0]).sum())
            counts[di] += n
            done += n
    np.savez_compressed(path, p=p, depths=np.array(depths), errors=errors,
                        shots=counts)
    return config, pi, si, time.perf_counter() - start


def _run_tasks(kind, task_fn, configs, args, depths, shots):
    out = Path(args.out) / kind / "parts"
    out.mkdir(parents=True, exist_ok=True)
    sequences = E.clifford_sequences(args.num_seq, 22, SEQ_SEED)
    p_indices = [int(i) for i in args.p_indices.split(",")]
    names = list(configs) if args.configs == "all" else args.configs.split(",")
    tasks = []
    for name in names:
        ci = list(configs).index(name)
        for pi in p_indices:
            for si in range(args.num_seq):
                path = out / f"{name}_p{pi}_s{si}.npz"
                if path.exists():
                    continue
                tasks.append((name, ci, pi, si, sequences[si], depths, shots,
                              str(path)))
    # Sequence-major order, so an interrupted run still covers every
    # (configuration, p) with the first sequences; slow high-p tasks first
    # within each sequence.
    tasks.sort(key=lambda t: (t[3], -t[2], t[0]))
    meta = {"kind": kind, "probs": PROBS, "depths": depths, "shots": shots,
            "num_seq": args.num_seq, "seq_seed": SEQ_SEED,
            "configs": {n: repr(configs[n]) for n in names},
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out.parent / f"meta_{int(time.time())}.json").write_text(
        json.dumps(meta, indent=1))
    print(f"{kind}: {len(tasks)} tasks to run", flush=True)
    start = time.perf_counter()
    deadline = start + 60.0 * args.max_minutes
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(task_fn, t) for t in tasks]
        for count, future in enumerate(as_completed(futures), 1):
            name, pi, si, seconds = future.result()
            print(f"[{time.perf_counter() - start:7.1f}s] {count}/{len(tasks)}"
                  f" {name} p{pi} s{si} ({seconds:.1f}s)", flush=True)
            if time.perf_counter() > deadline:
                print("deadline reached; cancelling remaining tasks",
                      flush=True)
                for f in futures:
                    f.cancel()
                break
    return 0


# ---------------------------------------------------------------------------
# First-order (single-fault) analysis
# ---------------------------------------------------------------------------


def _classify(probe, fault):
    loc = probe.locations[fault.location]
    kind = loc.kind
    if kind in ("cx_check", "cx_flag") and ER.is_hook(probe, fault):
        kind += "(hook)"
    if kind in ("meas_idle", "reset_idle", "meas_depol"):
        kind += "(data)" if loc.qubits[0] < C_NUM_DATA else "(anc)"
    return kind


C_NUM_DATA = 5


def _pauli_label(fault):
    names = {1: "X", 2: "Y", 3: "Z"}
    return "".join(f"{names[v]}{q}" for q, v in sorted(fault.paulis.items()))


def first_order_flags(sequence, depth, p_prior, flags, layout):
    probe = ER.build_instrumented_lrb(sequence, depth, p_prior, flags=flags,
                                      flag_layout=layout, mode="probe")
    dec_c = ER.build_instrumented_lrb(sequence, depth, p_prior, flags=flags,
                                      flag_layout=layout, mode="decode")
    sig = ER.probe_signatures(probe)
    n = dec_c.num_main_detectors + len(dec_c.flag_detectors)
    faults = ER.enumerate_single_faults(probe, sig, n)
    decoder = M.TransferMatrixMLD(dec_c.circuit.detector_error_model(),
                                  dec_c.decoder_slots())
    keys = sorted({f.detectors for f in faults if f.detectors})
    table = ER.first_order_table(faults, ER.decide_syndromes(decoder, keys, n))
    by_kind: dict[str, float] = {}
    by_class: dict[str, float] = {}
    total_kind: dict[str, float] = {}
    for f in faults:
        k = _classify(probe, f)
        total_kind[k] = total_kind.get(k, 0.0) + f.probability / p_prior
    for f in table["miscorrected"]:
        k = _classify(probe, f)
        by_kind[k] = by_kind.get(k, 0.0) + f.probability / p_prior
        loc = probe.locations[f.location]
        where = loc.gate if loc.kind.startswith("meas") else (
            f"{ER.C.STABILIZER_NAMES[loc.check[0]]}@{loc.check[1]}"
            if loc.check[0] >= 0 else loc.gate)
        label = f"{k} {where} {_pauli_label(f)}"
        by_class[label] = by_class.get(label, 0.0) + f.probability / p_prior
    hook_total = sum(f.probability for f in faults
                     if ER.is_hook(probe, f)) / p_prior
    hook_wrong = sum(f.probability for f in table["miscorrected"]
                     if ER.is_hook(probe, f)) / p_prior
    return {"depth": depth, "flags": flags, "layout": layout,
            "p_prior": p_prior,
            "c1": table["first_order_error"] / p_prior,
            "minority": table["minority_mass"] / p_prior,
            "undetected": table["undetected_logical"] / p_prior,
            "num_faults": len(faults), "num_syndromes": len(keys),
            "by_kind": by_kind, "total_by_kind": total_kind,
            "hook_total": hook_total, "hook_miscorrected": hook_wrong,
            "top_classes": sorted(by_class.items(), key=lambda kv: -kv[1])[:40]}


def first_order_erasure(sequence, depth, p_prior, spec):
    """First-order error of the herald-aware MLD and of the hybrid policy."""
    probe = ER.build_instrumented_lrb(sequence, depth, p_prior, erasure=spec,
                                      mode="probe")
    ec = ER.prepare_erasure_circuits(sequence, depth, p_prior, spec)
    sig = ER.probe_signatures(probe)
    n = ec.sample.num_main_detectors
    faults = ER.enumerate_single_faults(probe, sig, n, heralded_weight=True)
    keys = sorted({f.detectors for f in faults if f.detectors})
    table = ER.first_order_table(
        faults, ER.decide_syndromes(ec.mld_herald, keys, n))
    by_kind: dict[str, float] = {}
    for f in table["miscorrected"]:
        k = _classify(probe, f)
        by_kind[k] = by_kind.get(k, 0.0) + f.probability / p_prior
    detected = sum(f.probability for f in faults if f.detectors) / p_prior
    # Every single heralded erasure, all Paulis on its qubits.
    bad_locations = 0
    ambiguous_locations = 0
    herald_mass = 0.0
    for h, li in enumerate(ec.sample.herald_locations):
        loc = ec.sample.locations[li]
        factor = 16.0 / 15.0 if loc.channel == "DEPOLARIZE2" else 4.0 / 3.0
        herald_mass += factor * loc.herald_fraction * loc.probability
        events, flips = [], []
        for paulis in np.ndindex(*(4,) * len(loc.qubits)):
            dets, obs = ec.signatures.pauli(li, dict(zip(loc.qubits, paulis)))
            events.append(dets[:n])
            flips.append(obs)
        events = np.asarray(events)
        heralds = np.zeros((len(events), ec.sample.num_heralds), dtype=bool)
        heralds[:, h] = True
        llr = ec.mld_herald.log_likelihood_ratio(events, heralds)
        bad_locations += int(np.any((llr < 0) != np.asarray(flips)))
        e0, e1 = ec.explain.log_joint(events, heralds)
        ambiguous_locations += int(np.any(np.isfinite(e0) & np.isfinite(e1)))
    return {"depth": depth, "spec": [spec.two_qubit, spec.meas_idle],
            "p_prior": p_prior,
            "c1_mld_herald": table["first_order_error"] / p_prior,
            "by_kind": by_kind,
            "unheralded_detected_rate": detected,
            "herald_rate": herald_mass / p_prior,
            "num_herald_locations": ec.sample.num_heralds,
            "single_erasures_miscorrected": bad_locations,
            "single_erasures_ambiguous": ambiguous_locations}


def cmd_firstorder(args) -> int:
    out = Path(args.out) / "firstorder"
    out.mkdir(parents=True, exist_ok=True)
    sequences = E.clifford_sequences(args.num_seq, 22, SEQ_SEED)
    depths = [int(d) for d in args.depths.split(",")]
    results = {"flags": [], "erasure": []}
    start = time.perf_counter()
    jobs = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for si, seq in enumerate(sequences):
            for depth in depths:
                for name, (flags, layout) in FLAG_CONFIGS.items():
                    for p_prior in (1e-4, 5e-3):
                        jobs.append(("flags", si, name, pool.submit(
                            first_order_flags, seq, depth, p_prior, flags,
                            layout)))
                for name, spec in ERASURE_CONFIGS.items():
                    jobs.append(("erasure", si, name, pool.submit(
                        first_order_erasure, seq, depth, 1e-4, spec)))
        for kind, si, name, future in jobs:
            row = future.result()
            row["seq"] = si
            row["config"] = name
            results[kind].append(row)
            print(f"[{time.perf_counter() - start:7.1f}s] {kind} {name} "
                  f"s{si} d{row['depth']}", flush=True)
    (out / "firstorder.json").write_text(json.dumps(results, indent=1))
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def _load_parts(directory: Path, prefix: str, num_seq: int):
    """Stack per-sequence parts into CountTables (missing parts -> None)."""
    files = [directory / f"{prefix}_s{si}.npz" for si in range(num_seq)]
    present = [f for f in files if f.exists()]
    if not present:
        return None
    datas = [np.load(f) for f in present]
    names = sorted({k.split("__")[0] for k in datas[0].files
                    if k.endswith("__shots")})
    tables = {}
    for name in names:
        tables[name] = B.CountTable(
            np.stack([d[f"{name}__shots"] for d in datas]),
            np.stack([d[f"{name}__accepted"] for d in datas]),
            np.stack([d[f"{name}__errors"] for d in datas]))
    return {"tables": tables, "datas": datas, "depths":
            [int(x) for x in datas[0]["depths"]], "n": len(present)}


def fit_decay(depths, fid_by_seq, min_points=3):
    """Fit F(m) = b f^m (offset 0) to the sequence-mean fidelity.

    Weighted least squares on the mean over sequences (weights from the
    standard error across sequences), grid scan over f as in the repo;
    points with mean F >= 1 or <= 0.02 are dropped. The uncertainty is the
    standard deviation of f over 200 bootstrap resamples of the sequences.
    """
    depths = np.asarray(depths, float)
    grid = np.linspace(0.5, 1.0, 5001)

    def fit(sample):
        with np.errstate(invalid="ignore"), _quiet():
            mean = np.nanmean(sample, axis=0)
            se = np.nanstd(sample, axis=0) / np.sqrt(sample.shape[0]) + 1e-4
        keep = (mean < 1.0) & (mean > 0.02) & np.isfinite(mean)
        if keep.sum() < min_points:
            return np.nan, np.nan
        m, y, w = depths[keep], mean[keep], 1.0 / se[keep] ** 2
        g = grid[:, None] ** m[None, :]
        b = (g * (w * y)).sum(axis=1) / (g * g * w).sum(axis=1)
        chi = (w * (y[None, :] - b[:, None] * g) ** 2).sum(axis=1)
        k = int(np.argmin(chi))
        return grid[k], b[k]

    f0, b0 = fit(fid_by_seq)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(200):
        idx = rng.integers(0, fid_by_seq.shape[0], fid_by_seq.shape[0])
        boots.append(fit(fid_by_seq[idx])[0])
    with _quiet():
        return f0, b0, float(np.nanstd(boots))


class _quiet:
    """Silence numpy's empty-slice RuntimeWarnings inside a block."""

    def __enter__(self):
        import warnings
        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("ignore", category=RuntimeWarning)

    def __exit__(self, *exc):
        self._ctx.__exit__(*exc)
        return False


def _fmt(mean, err, digits=4):
    if not np.isfinite(mean):
        return "n/a"
    return f"{mean:.{digits}f}({int(round(err * 10 ** digits)):d})" \
        if err * 10 ** digits < 99.5 else f"{mean:.{digits}f}+-{err:.{digits}f}"


def _summ(table: B.CountTable):
    fid = table.fidelity()
    acc = table.accepted / np.maximum(table.shots, 1)
    n = fid.shape[0]
    with np.errstate(invalid="ignore"), _quiet():
        f_mean = np.nanmean(fid, axis=0)
        f_se = np.nanstd(fid, axis=0) / np.sqrt(np.maximum(
            np.sum(np.isfinite(fid), axis=0), 1))
        a_mean = acc.mean(axis=0)
        a_se = acc.std(axis=0) / np.sqrt(n)
    return f_mean, f_se, a_mean, a_se


def cmd_report(args) -> int:
    out = Path(args.out)
    lines = ["# WP I result tables (generated by erasure_leakage.py report)",
             ""]
    csv_rows = []
    # -------- erasure
    edir = out / "erasure" / "parts"
    if edir.exists():
        lines += ["## Erasure conversion", ""]
        for pi, p in enumerate(PROBS):
            for name in ERASURE_CONFIGS:
                part = _load_parts(edir, f"{name}_p{pi}", args.num_seq)
                if part is None:
                    continue
                depths = part["depths"]
                lines += [f"### p = {p:g}, config {name} "
                          f"({part['n']} sequences)", "",
                          "| method | " + " | ".join(f"m={d}" for d in depths)
                          + " | f (fit) |",
                          "|---|" + "---|" * (len(depths) + 1)]
                for method in ER.ERASURE_METHODS:
                    t = part["tables"].get(method)
                    if t is None:
                        continue
                    f_mean, f_se, a_mean, a_se = _summ(t)
                    f_fit, b_fit, f_err = fit_decay(depths, t.fidelity())
                    lines.append(f"| F {method} | " + " | ".join(
                        _fmt(a, b) for a, b in zip(f_mean, f_se))
                        + f" | {_fmt(f_fit, f_err)} |")
                    if method not in ("mld", "mld_herald"):
                        lines.append(f"| acc {method} | " + " | ".join(
                            _fmt(a, b) for a, b in zip(a_mean, a_se)) + " | |")
                    for di, d in enumerate(depths):
                        csv_rows.append(("erasure", name, p, method, d,
                                         f_mean[di], f_se[di], a_mean[di],
                                         a_se[di], f_fit, f_err))
                lines.append("")
    # -------- leakage
    ldir = out / "leakage" / "parts"
    if ldir.exists():
        lines += ["## Leakage", ""]
        for pi, p in enumerate(PROBS):
            for name in LEAKAGE_CONFIGS:
                part = _load_parts(ldir, f"{name}_p{pi}", args.num_seq)
                if part is None:
                    continue
                depths = part["depths"]
                lines += [f"### p = {p:g}, config {name} "
                          f"({part['n']} sequences)", "",
                          "| method | " + " | ".join(f"m={d}" for d in depths)
                          + " | f (fit) |",
                          "|---|" + "---|" * (len(depths) + 1)]
                for method in L.LEAKAGE_METHODS:
                    t = part["tables"][method]
                    f_mean, f_se, a_mean, a_se = _summ(t)
                    f_fit, b_fit, f_err = fit_decay(depths, t.fidelity())
                    lines.append(f"| F {method} | " + " | ".join(
                        _fmt(a, b) for a, b in zip(f_mean, f_se))
                        + f" | {_fmt(f_fit, f_err)} |")
                    if method != "mld":
                        lines.append(f"| acc {method} | " + " | ".join(
                            _fmt(a, b) for a, b in zip(a_mean, a_se)) + " | |")
                    for di, d in enumerate(depths):
                        csv_rows.append(("leakage", name, p, method, d,
                                         f_mean[di], f_se[di], a_mean[di],
                                         a_se[di], f_fit, f_err))
                lines.append("")
    # -------- flags (sampled)
    fdir = out / "flags" / "parts"
    if fdir.exists():
        lines += ["## Flags (sampled exact-MLD logical error rate)", ""]
        for pi, p in enumerate(PROBS):
            header = None
            for name in FLAG_CONFIGS:
                files = sorted(fdir.glob(f"{name}_p{pi}_s*.npz"))
                if not files:
                    continue
                datas = [np.load(f) for f in files]
                depths = [int(x) for x in datas[0]["depths"]]
                err = np.stack([d["errors"] for d in datas]).sum(axis=0)
                shots = np.stack([d["shots"] for d in datas]).sum(axis=0)
                rate = err / shots
                se = np.sqrt(rate * (1 - rate) / shots)
                if header is None:
                    lines += [f"### p = {p:g}", "", "| config | "
                              + " | ".join(f"m={d}" for d in depths) + " |",
                              "|---|" + "---|" * len(depths)]
                    header = True
                lines.append(f"| {name} ({len(files)} seq) | " + " | ".join(
                    _fmt(a, b, 5) for a, b in zip(rate, se)) + " |")
                for di, d in enumerate(depths):
                    csv_rows.append(("flags", name, p, "mld_error", d,
                                     rate[di], se[di], 1.0, 0.0, np.nan,
                                     np.nan))
            lines.append("")
    (out / "tables.md").write_text("\n".join(lines) + "\n")
    with open(out / "summary.csv", "w") as handle:
        handle.write("study,config,p,method,depth,value_mean,value_se,"
                     "acceptance_mean,acceptance_se,fit_f,fit_f_err\n")
        for row in csv_rows:
            handle.write(",".join(str(x) for x in row) + "\n")
    print(f"wrote {out / 'tables.md'} and summary.csv ({len(csv_rows)} rows)")
    return 0


def _paired(a: np.ndarray, b: np.ndarray):
    """Mean and standard error over sequences of a - b (NaNs dropped)."""
    d = a - b
    with _quiet():
        n = np.sum(np.isfinite(d), axis=0)
        return np.nanmean(d, axis=0), np.nanstd(d, axis=0) / np.sqrt(
            np.maximum(n, 1))


def _gap_curve(hist: np.ndarray, edges: np.ndarray, threshold_index: int):
    """Per-sequence (accepted, errors) for gap >= edges[threshold_index].

    ``hist`` has shape (seq, depth, bins, 2) with [..., 0] correct and
    [..., 1] wrong decisions.
    """
    kept = hist[:, :, threshold_index:, :].sum(axis=2)
    return kept.sum(axis=2), kept[..., 1]


def _early_late(depths, fid):
    depths = np.asarray(depths)
    early = depths <= 10
    late = depths >= 10
    fe = fit_decay(depths[early], fid[:, early])
    fl = fit_decay(depths[late], fid[:, late])
    return fe, fl


def cmd_highlights(args) -> dict:
    """Curated comparisons for the notes (written to highlights.json/.md)."""
    out = Path(args.out)
    hl: dict = {"erasure": {}, "leakage": {}, "flags": {}, "firstorder": {}}
    md = ["# WP I highlights (generated by erasure_leakage.py report)", ""]
    # ---------------- erasure
    edir = out / "erasure" / "parts"
    if edir.exists():
        md += ["## Erasure: full correction, hybrid and gap postselection", "",
               "F = mean over sequences (SE); dF = paired difference "
               "mld_herald - mld on identical shots; acceptance columns are "
               "fractions of all shots. `gap@LRBD` = herald-aware MLD with "
               "|LLR| threshold chosen so its pooled acceptance is the "
               "smallest one >= the LRB-D (unif_1_final) acceptance.", ""]
        for pi, p in enumerate(PROBS):
            for name in ERASURE_CONFIGS:
                part = _load_parts(edir, f"{name}_p{pi}", args.num_seq)
                if part is None:
                    continue
                depths = part["depths"]
                t = part["tables"]
                edges = part["datas"][0]["gap_edges"]
                hist_h = np.stack([d["gap__mld_herald"] for d in part["datas"]])
                fm, fm_se, _, _ = _summ(t["mld"])
                fh, fh_se, _, _ = _summ(t["mld_herald"])
                dF, dF_se = _paired(t["mld_herald"].fidelity(),
                                    t["mld"].fidelity())
                fit_m = fit_decay(depths, t["mld"].fidelity())
                fit_h = fit_decay(depths, t["mld_herald"].fidelity())
                rows = []
                for di, d in enumerate(depths):
                    row = {"depth": d, "F_mld": [fm[di], fm_se[di]],
                           "F_mld_herald": [fh[di], fh_se[di]],
                           "dF": [dF[di], dF_se[di]]}
                    for method in ("lrbd_unif_1", "lrbd_unif_1_final",
                                   "lrbd_unif_1_final_noherald", "hybrid",
                                   "lrbd_const_1"):
                        f_, fse, a_, ase = _summ(t[method])
                        row[method] = {"F": [f_[di], fse[di]],
                                       "acc": [a_[di], ase[di]]}
                    # gap postselection matched to LRB-D acceptance
                    target = row["lrbd_unif_1_final"]["acc"][0]
                    shots = t["mld_herald"].shots[:, di].astype(float)
                    best = None
                    for k in range(len(edges) - 1):
                        acc_k, err_k = _gap_curve(hist_h[:, di:di + 1],
                                                  edges, k)
                        acc_frac = acc_k[:, 0].sum() / shots.sum()
                        if acc_frac >= target:
                            best = (k, acc_k[:, 0], err_k[:, 0], acc_frac)
                        else:
                            break
                    if best is not None:
                        k, acc_k, err_k, acc_frac = best
                        with np.errstate(invalid="ignore", divide="ignore"):
                            fid_k = np.abs(1 - 2 * err_k / acc_k)
                        with _quiet():
                            row["gap_at_lrbd"] = {
                                "threshold": float(edges[k]),
                                "acc": float(acc_frac),
                                "F": [float(np.nanmean(fid_k)), float(
                                    np.nanstd(fid_k) / np.sqrt(
                                        max(np.isfinite(fid_k).sum(), 1)))]}
                    for g in (2.0, 4.0, 6.0):
                        k = int(np.searchsorted(edges, g))
                        acc_k, err_k = _gap_curve(hist_h[:, di:di + 1], edges,
                                                  k)
                        with np.errstate(invalid="ignore", divide="ignore"):
                            fid_k = np.abs(1 - 2 * err_k[:, 0] / acc_k[:, 0])
                        with _quiet():
                            row[f"gap{g:g}"] = {
                                "acc": float(acc_k[:, 0].sum() / shots.sum()),
                                "F": [float(np.nanmean(fid_k)), float(
                                    np.nanstd(fid_k) / np.sqrt(
                                        max(np.isfinite(fid_k).sum(), 1)))]}
                    hs = part["datas"]
                    stats = np.stack([d["herald_stats"] for d in hs])[:, di]
                    row["heralds_per_shot"] = float(stats[:, 2].sum()
                                                    / stats[:, 0].sum())
                    row["frac_shots_with_herald"] = float(
                        stats[:, 1].sum() / stats[:, 0].sum())
                    row["hybrid_ambiguous_frac"] = float(
                        np.stack([d["ambiguous"] for d in hs])[:, di].sum()
                        / stats[:, 0].sum())
                    rows.append(row)
                key = f"{name}|{p:g}"
                hl["erasure"][key] = {"n_seq": part["n"], "rows": rows,
                                      "fit_mld": list(fit_m),
                                      "fit_mld_herald": list(fit_h)}
                md += [f"### p = {p:g}, {name} ({part['n']} seq); fitted f: "
                       f"mld {_fmt(fit_m[0], fit_m[2])}, mld_herald "
                       f"{_fmt(fit_h[0], fit_h[2])}", "",
                       "| m | F mld | F mld_herald | dF | acc LRB-D(u1f) | "
                       "F LRB-D(u1f) | acc hybrid | F hybrid | acc u1f-noherald"
                       " | gap@LRBD thr | F gap@LRBD |",
                       "|---|---|---|---|---|---|---|---|---|---|---|"]
                for row in rows:
                    g = row.get("gap_at_lrbd", {"threshold": np.nan, "F":
                                                [np.nan, 0]})
                    md.append(
                        f"| {row['depth']} | {_fmt(*row['F_mld'])} | "
                        f"{_fmt(*row['F_mld_herald'])} | {_fmt(*row['dF'])} | "
                        f"{_fmt(*row['lrbd_unif_1_final']['acc'])} | "
                        f"{_fmt(*row['lrbd_unif_1_final']['F'])} | "
                        f"{_fmt(*row['hybrid']['acc'])} | "
                        f"{_fmt(*row['hybrid']['F'])} | "
                        f"{_fmt(*row['lrbd_unif_1_final_noherald']['acc'])} | "
                        f"{g['threshold']:g} | {_fmt(*g['F'])} |")
                md.append("")
    # ---------------- leakage
    ldir = out / "leakage" / "parts"
    if ldir.exists():
        md += ["## Leakage", ""]
        for pi, p in enumerate(PROBS):
            base = _load_parts(ldir, f"none_p{pi}", args.num_seq)
            for name in LEAKAGE_CONFIGS:
                part = _load_parts(ldir, f"{name}_p{pi}", args.num_seq)
                if part is None:
                    continue
                depths = part["depths"]
                t = part["tables"]
                entry = {"n_seq": part["n"], "depths": depths}
                for method in L.LEAKAGE_METHODS:
                    f_, fse, a_, ase = _summ(t[method])
                    entry[method] = {"F": [list(f_), list(fse)],
                                     "acc": [list(a_), list(ase)],
                                     "fit": list(fit_decay(
                                         depths, t[method].fidelity()))}
                (fe, fl) = _early_late(depths, t["mld"].fidelity())
                entry["mld_fit_early"] = list(fe)
                entry["mld_fit_late"] = list(fl)
                if base is not None:
                    fb, fb_se, ab, ab_se = _summ(base["tables"]["mld"])
                    f_, fse, _, _ = _summ(t["mld"])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        r = f_ / fb
                        r_se = np.abs(r) * np.sqrt((fse / f_) ** 2
                                                   + (fb_se / fb) ** 2)
                    entry["mld_ratio_to_none"] = [list(r), list(r_se)]
                    fu, fuse, au, ause = _summ(t["lrbd_unif_1_final"])
                    fbu, fbuse, abu, abuse = _summ(
                        base["tables"]["lrbd_unif_1_final"])
                    with np.errstate(invalid="ignore", divide="ignore"):
                        entry["u1f_acc_ratio_to_none"] = list(au / abu)
                datas = part["datas"]
                entry["leaked_trajectory"] = list(np.mean(
                    [d["leaked_trajectory"] for d in datas], axis=0))
                entry["autocorr"] = np.mean([d["autocorr"] for d in datas],
                                            axis=0).mean(axis=1).tolist()
                entry["flagged_frac"] = list(
                    np.stack([d["flagged"] for d in datas]).sum(axis=0)
                    / t["mld"].shots.sum(axis=0))
                hl["leakage"][f"{name}|{p:g}"] = entry
                md += [f"### p = {p:g}, leakage {name} ({part['n']} seq)", "",
                       "fitted f (all / m<=10 / m>=10): mld "
                       f"{_fmt(entry['mld']['fit'][0], entry['mld']['fit'][2])}"
                       f" / {_fmt(fe[0], fe[2])} / {_fmt(fl[0], fl[2])}; "
                       f"lag-1..6 detection autocorrelation "
                       + ", ".join(f"{x:.4f}" for x in entry["autocorr"][1:]),
                       "", "| m | F mld | F mld_flag | acc mld_flag | "
                       "F/F_none (mld) | F LRB-D(u1f) | acc LRB-D(u1f) | "
                       "acc u1f_flag | F const1 | F const1_flag | "
                       "acc const1_flag |",
                       "|---|---|---|---|---|---|---|---|---|---|---|"]
                for di, d in enumerate(depths):
                    def cell(method, what):
                        v = entry[method][what]
                        return _fmt(v[0][di], v[1][di])
                    ratio_cell = "n/a"
                    if "mld_ratio_to_none" in entry:
                        rr = entry["mld_ratio_to_none"]
                        ratio_cell = _fmt(rr[0][di], rr[1][di])
                    md.append(
                        f"| {d} | {cell('mld', 'F')} | {cell('mld_flag', 'F')}"
                        f" | {cell('mld_flag', 'acc')} | {ratio_cell} | "
                        f"{cell('lrbd_unif_1_final', 'F')} | "
                        f"{cell('lrbd_unif_1_final', 'acc')} | "
                        f"{cell('lrbd_unif_1_final_flag', 'acc')} | "
                        f"{cell('lrbd_const_1', 'F')} | "
                        f"{cell('lrbd_const_1_flag', 'F')} | "
                        f"{cell('lrbd_const_1_flag', 'acc')} |")
                md += ["", "leaked-data population before each reset layer "
                       "(m=22): " + ", ".join(
                           f"{x:.3f}" for x in entry["leaked_trajectory"][
                               :max(depths) + 1]), ""]
    # ---------------- flags (sampled)
    fdir = out / "flags" / "parts"
    if fdir.exists():
        for pi, p in enumerate(PROBS):
            for name in FLAG_CONFIGS:
                files = sorted(fdir.glob(f"{name}_p{pi}_s*.npz"))
                if not files:
                    continue
                datas = [np.load(f) for f in files]
                err = np.stack([d["errors"] for d in datas]).sum(axis=0)
                shots = np.stack([d["shots"] for d in datas]).sum(axis=0)
                hl["flags"][f"{name}|{p:g}"] = {
                    "depths": [int(x) for x in datas[0]["depths"]],
                    "rate": (err / shots).tolist(), "shots": shots.tolist(),
                    "n_seq": len(files)}
    # ---------------- first order
    fo = out / "firstorder" / "firstorder.json"
    if fo.exists():
        data = json.loads(fo.read_text())
        md += ["## First-order (single-fault) analysis", ""]
        flags_rows: dict = {}
        for row in data["flags"]:
            key = (row["config"], row["depth"], row["p_prior"])
            flags_rows.setdefault(key, []).append(row)
        md += ["c1 = first-order exact-MLD logical error / p (mean over "
               "sequences, prior p in parentheses); hook = mass of hook "
               "faults miscorrected / p.", "",
               "| config | depth | prior | c1 | hook miscorrected | "
               "data-idle share |", "|---|---|---|---|---|---|"]
        summary = {}
        for (cfg, depth, prior), rows in sorted(flags_rows.items(),
                                                key=lambda kv: (kv[0][1],
                                                                kv[0][2],
                                                                kv[0][0])):
            c1 = np.mean([r["c1"] for r in rows])
            c1_sd = np.std([r["c1"] for r in rows])
            hook = np.mean([r["hook_miscorrected"] for r in rows])
            idle = np.mean([(r["by_kind"].get("meas_idle(data)", 0)
                             + r["by_kind"].get("reset_idle(data)", 0))
                            / max(r["c1"], 1e-12) for r in rows])
            summary[f"{cfg}|{depth}|{prior:g}"] = {
                "c1": c1, "c1_sd": c1_sd, "hook": hook, "idle_share": idle,
                "by_kind": {k: float(np.mean([r["by_kind"].get(k, 0.0)
                                               for r in rows]))
                            for k in sorted({k for r in rows
                                             for k in r["by_kind"]})}}
            md.append(f"| {cfg} | {depth} | {prior:g} | {c1:.3f} "
                      f"(sd {c1_sd:.3f}) | {hook:.3f} | {idle:.3f} |")
        hl["firstorder"]["flags"] = summary
        er_rows: dict = {}
        for row in data["erasure"]:
            er_rows.setdefault((row["config"], row["depth"]), []).append(row)
        md += ["", "| erasure config | depth | c1 herald-aware MLD | "
               "unheralded detected rate / p | herald rate / p | "
               "single erasures miscorrected | ambiguous |",
               "|---|---|---|---|---|---|---|"]
        esum = {}
        for (cfg, depth), rows in sorted(er_rows.items(),
                                         key=lambda kv: (kv[0][1], kv[0][0])):
            vals = {k: float(np.mean([r[k] for r in rows])) for k in (
                "c1_mld_herald", "unheralded_detected_rate", "herald_rate")}
            bad = int(sum(r["single_erasures_miscorrected"] for r in rows))
            amb = int(sum(r["single_erasures_ambiguous"] for r in rows))
            locs = int(sum(r["num_herald_locations"] for r in rows))
            esum[f"{cfg}|{depth}"] = dict(vals, miscorrected=bad,
                                          ambiguous=amb, locations=locs)
            md.append(f"| {cfg} | {depth} | {vals['c1_mld_herald']:.3f} | "
                      f"{vals['unheralded_detected_rate']:.2f} | "
                      f"{vals['herald_rate']:.2f} | {bad}/{locs} | {amb} |")
        hl["firstorder"]["erasure"] = esum
    (out / "highlights.md").write_text("\n".join(md) + "\n")

    def clean(obj):
        if isinstance(obj, dict):
            return {k: clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [clean(v) for v in obj]
        if isinstance(obj, (np.floating, float)):
            return None if not np.isfinite(obj) else float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        return obj

    (out / "highlights.json").write_text(json.dumps(clean(hl), indent=1))
    print(f"wrote {out / 'highlights.md'} and highlights.json")
    return hl


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["erasure", "leakage", "flags",
                                            "firstorder", "report"])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-seq", type=int, default=20)
    parser.add_argument("--p-indices", default="0,1,2")
    parser.add_argument("--configs", default="all")
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--max-minutes", type=float, default=18.0)
    args = parser.parse_args()
    if args.command == "erasure":
        depths = ([int(d) for d in args.depths.split(",")] if args.depths
                  else DEPTHS)
        return _run_tasks("erasure", _erasure_task, ERASURE_CONFIGS, args,
                          depths, args.shots or 40_000)
    if args.command == "leakage":
        depths = ([int(d) for d in args.depths.split(",")] if args.depths
                  else DEPTHS)
        return _run_tasks("leakage", _leakage_task, LEAKAGE_CONFIGS, args,
                          depths, args.shots or 50_000)
    if args.command == "flags":
        depths = ([int(d) for d in args.depths.split(",")] if args.depths
                  else [0, 2, 4, 6, 10])
        return _run_tasks("flags", _flag_task, FLAG_CONFIGS, args, depths,
                          args.shots or 20_000)
    if args.command == "firstorder":
        if args.depths is None:
            args.depths = "0,2,4,6,10"
        return cmd_firstorder(args)
    cmd_report(args)
    cmd_highlights(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
