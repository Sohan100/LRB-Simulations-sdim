"""Work package B: matching (MWPM) decoders against exact MLD on identical shots.

For every (p, Clifford sequence, depth) the ``data5`` LRB circuit is sampled
once and every decoder of ``lrb_qubit.decoders_matching.DecoderSuite``
decodes the same detection events:

``mld``              exact maximum likelihood (``mld.TransferMatrixMLD``), gap |LLR|
``mwe``              exact minimum-weight error on the undecomposed hypergraph
``mwpm``             PyMatching 2.3.1, plain, decomposed DEM
``mwpm_corr``        PyMatching correlated matching (stripped DEM, see module)
``mwpm_gap``         exact class weights on PyMatching's own graph (complementary gap)
``mwpm_classaware``  exact class weights on the class-aware graph
``mwpm_obs``         observable as detector, decoded with it forced to 0 and 1
``mwpm_obs_corr``    the same with correlated matching (heuristic gap)
``none``             always predict "no flip"

The parent pools the sequences of each (p, depth), computes logical error
rates, agreement with MLD and gap-thresholded acceptance-vs-error curves, and
attaches leave-one-sequence-out jackknife standard errors. It also writes the
decomposition census and the exact single-fault (first-order) table.

Each finished task is pickled to ``--raw-dir`` at once (resumable; existing
files with the same shots and seed are reused), so a grid can be split over
several runs and merged with ``--summarize-only``. Seeds depend on the value
of p (not its list position), so split runs reproduce a single run exactly.

Example (login node, small):

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python studies/correction-decoders/qubit/scripts/matching_vs_mld.py --out /tmp/b --raw-dir /tmp/b/raw \
        --num-seq 2 --shots 2000 --workers 2 --p-list 0.002 --depths 0,2
"""

from __future__ import annotations

import os
import sys

# One BLAS thread per process; must be set before the interpreter starts
# (a site hook imports numpy at startup), so re-exec once.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import pickle  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import decoders_matching as D  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

DECODERS = ("mld", "mwe", "mwpm", "mwpm_corr", "mwpm_gap", "mwpm_classaware",
            "mwpm_obs", "mwpm_obs_corr", "none")
SOFT = ("mld", "mwe", "mwpm_gap", "mwpm_classaware", "mwpm_obs",
        "mwpm_obs_corr")
# Gap histogram: first bin [0, 1e-9) holds exact ties.
GAP_EDGES = np.concatenate([[0.0, D.TIE_TOL], np.arange(0.02, 60.0 + 1e-9, 0.02),
                            [np.inf]])
CALIB_EDGES = np.array([0.0, D.TIE_TOL, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0,
                        12.0, np.inf])
TARGET_ACCEPTANCE = (1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.9, 0.8, 0.7, 0.5,
                     0.3, 0.2, 0.1)
FIRST_ORDER_P = 1e-5
GRID_P = (0.0005, 0.002, 0.005, 0.01)
DEFAULT_OUT = ROOT / "studies/correction-decoders/qubit/results/B"


# ---------------------------------------------------------------------------
# Worker tasks
# ---------------------------------------------------------------------------

def _hist(gap: np.ndarray, wrong: np.ndarray, edges: np.ndarray) -> np.ndarray:
    bins = np.clip(np.searchsorted(edges, gap, side="right") - 1, 0,
                   len(edges) - 2)
    out = np.zeros((len(edges) - 1, 2), dtype=np.int64)
    np.add.at(out, (bins, wrong.astype(np.int64)), 1)
    return out


def seed_for(base: int, p: float, seq_index: int, depth: int) -> int:
    """Sampler seed from the value of p, the sequence and the depth."""
    return base + 1_000_000 * int(round(p * 1e5)) + 1000 * seq_index + depth


def sample_task(args) -> dict:
    """Sample one circuit and decode it with every decoder."""
    p_index, p, seq_index, sequence, d_index, depth, shots, seed = args
    start = time.perf_counter()
    lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
    suite = D.DecoderSuite.build(lc)
    sampler = lc.circuit.compile_detector_sampler(seed=seed)
    out = {"key": (p_index, seq_index, d_index), "shots": 0,
           "errors": {k: 0 for k in DECODERS},
           "agree": {k: 0 for k in DECODERS},
           "wrong_mldright": {k: 0 for k in DECODERS},
           "right_mldwrong": {k: 0 for k in DECODERS},
           "ties": {k: 0 for k in SOFT}, "tie_errors": {k: 0 for k in SOFT},
           "mld_errors_on_ties": {k: 0 for k in SOFT},
           "hist": {k: np.zeros((len(GAP_EDGES) - 1, 2), np.int64)
                    for k in SOFT},
           "calib": {k: np.zeros((len(CALIB_EDGES) - 1, 3)) for k in SOFT},
           "zero_syndrome": [0, 0],
           "pm_argmin_disagree": 0, "pm_weight_maxdiff": 0.0}
    done = 0
    batch = 50_000
    while done < shots:
        n = min(batch, shots - done)
        dets, obs = sampler.sample(n, separate_observables=True)
        flip = obs[:, 0]
        pred, gaps, extras = suite.run(dets)
        mld_wrong = pred["mld"] != flip
        for name in DECODERS:
            wrong = pred[name] != flip
            out["errors"][name] += int(wrong.sum())
            out["agree"][name] += int((pred[name] == pred["mld"]).sum())
            out["wrong_mldright"][name] += int((wrong & ~mld_wrong).sum())
            out["right_mldwrong"][name] += int((~wrong & mld_wrong).sum())
        for name in SOFT:
            wrong = pred[name] != flip
            tie = gaps[name] < D.TIE_TOL
            out["ties"][name] += int(tie.sum())
            out["tie_errors"][name] += int((tie & wrong).sum())
            out["mld_errors_on_ties"][name] += int((tie & mld_wrong).sum())
            out["hist"][name] += _hist(gaps[name], wrong, GAP_EDGES)
            bins = np.clip(np.searchsorted(CALIB_EDGES, gaps[name],
                                           side="right") - 1, 0,
                           len(CALIB_EDGES) - 2)
            predicted = 1.0 / (1.0 + np.exp(np.minimum(gaps[name], 700.0)))
            np.add.at(out["calib"][name], (bins, 0), 1.0)
            np.add.at(out["calib"][name], (bins, 1), wrong.astype(float))
            np.add.at(out["calib"][name], (bins, 2), predicted)
        clean = ~dets.any(axis=1)
        out["zero_syndrome"][0] += int(clean.sum())
        out["zero_syndrome"][1] += int((clean & flip).sum())
        out["pm_argmin_disagree"] += int((pred["mwpm"]
                                          != pred["mwpm_gap"]).sum())
        weights = extras["mwpm_gap_weights"].min(axis=1)
        out["pm_weight_maxdiff"] = max(
            out["pm_weight_maxdiff"],
            float(np.max(np.abs(weights - extras["mwpm_weight"]))))
        out["shots"] += n
        done += n
    out["seconds"] = time.perf_counter() - start
    return out


def structure_task(args) -> dict:
    """Decomposition census and exact single-fault table for one circuit."""
    seq_index, sequence, depth = args
    p_list = GRID_P
    rows = []
    lc_ref = Q.build_lrb_circuit(sequence, depth, 2e-3, terminal="data5")
    census = D.decomposition_census(lc_ref.circuit).as_row()
    census.update(seq=seq_index, depth=depth, p_ref=2e-3)
    for p in [FIRST_ORDER_P] + list(p_list):
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
        suite = D.DecoderSuite.build(lc)
        table = D.single_fault_table(suite)
        labels = D.pymatching_l_attribution(
            lc.circuit.detector_error_model(decompose_errors=True))
        mechs = [m for m in M.dem_mechanisms(lc.circuit.detector_error_model())
                 if m.probability > 0]
        prob = table["probability"]
        row = {"seq": seq_index, "depth": depth, "p": p,
               "single_fault_mass": prob.sum() / p,
               "mld_tie_mass": (prob * table["mld_tie"]).sum() / p}
        for name in DECODERS:
            row["c_" + name] = (prob * table["wrong_" + name]).sum() / p
        # Attribution of plain-MWPM excess over MLD.
        excess: dict[str, float] = {}
        for i, mech in enumerate(mechs):
            if table["wrong_mwpm"][i] == table["wrong_mld"][i]:
                continue
            if table["wrong_mld"][i]:
                label = "mld_wrong_mwpm_right"
            else:
                size = "1" if len(mech.detectors) == 1 else (
                    "2" if len(mech.detectors) == 2 else "3+")
                label = labels.get((mech.detectors, mech.flips_observable),
                                   "missing") + "_size" + size
            excess[label] = excess.get(label, 0.0) + prob[i] / p
        for label, value in excess.items():
            row["mwpm_attr_" + label] = value
        rows.append(row)
    return {"census": census, "first_order": rows}


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def jackknife(values: np.ndarray, statistic) -> tuple[float, float]:
    """Statistic on all units and its leave-one-unit-out jackknife SE.

    ``values`` has the sequence index on axis 0; ``statistic`` maps an array
    of that shape (summed over axis 0 inside) to a number.
    """
    k = values.shape[0]
    full = statistic(values.sum(axis=0))
    if k < 2:
        return full, float("nan")
    total = values.sum(axis=0)
    loo = np.array([statistic(total - values[i]) for i in range(k)])
    loo = loo[np.isfinite(loo)]
    if len(loo) < 2:
        return full, float("nan")
    se = np.sqrt((k - 1) / k * np.sum((loo - loo.mean()) ** 2))
    return full, float(se)


def error_at_acceptance(hist: np.ndarray, target: float
                        ) -> tuple[float, float, float]:
    """Accepted error rate at acceptance ``target`` (gap threshold rule).

    ``hist[b] = (correct, wrong)`` per gap bin. Shots are accepted from the
    largest gap down; the bin where the target is crossed is accepted
    partially (a randomized threshold), which interpolates linearly.

    Returns:
        ``(acceptance, accepted_wrong, accepted_shots)`` (the achieved
        acceptance equals ``target`` unless ``target`` exceeds 1).
    """
    counts = hist.sum(axis=1)[::-1].astype(float)
    wrong = hist[:, 1][::-1].astype(float)
    total = counts.sum()
    if total == 0:
        return float("nan"), float("nan"), 0.0
    need = min(target, 1.0) * total
    cum = np.cumsum(counts)
    k = int(np.searchsorted(cum, need - 1e-9))
    k = min(k, len(cum) - 1)
    before = cum[k - 1] if k > 0 else 0.0
    wrong_before = wrong[:k].sum()
    frac = 0.0 if counts[k] == 0 else (need - before) / counts[k]
    accepted_wrong = wrong_before + frac * wrong[k]
    return need / total, accepted_wrong, need


def _ratio(num: float, den: float) -> float:
    return num / den if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k, "")) for k in keys})


def _fmt(value):
    if isinstance(value, float):
        return f"{value:.6g}"
    return value


def physical_rb_fidelity(sequence, depth: int, p_rb: float) -> float:
    """Exact <Z> of ``circuits.build_physical_rb_circuit(sequence, depth, p_rb)``.

    Every gate token is followed by ``DEPOLARIZE1(q)`` with ``q = 0.1 p_rb``.
    A depolarizing channel commutes with Clifford conjugation, so the final
    Z readout sees ``N`` independent depolarizing channels and
    ``<Z> = (1 - 4q/3)^N``, with ``N`` the number of gate tokens in the
    forward and inverse halves.
    """
    tokens = sum(len(C.clifford_gate_sequence(d))
                 + len(C.clifford_inverse_sequence(d))
                 for d in sequence[:depth])
    return (1.0 - 4.0 * 0.1 * p_rb / 3.0) ** tokens


def physical_rb_comparison(out: Path) -> None:
    """Decoded LRB fidelity against physical RB at p_RB = p and p_RB = 5p.

    Reads ``meta.json`` and ``per_circuit.csv`` from ``out`` and writes
    ``physical_rb_comparison.csv``. Fidelities follow the repo convention:
    ``F = |1 - 2 errors/shots|`` per sequence, averaged over sequences, with
    the standard error of that mean.
    """
    meta = json.loads((out / "meta.json").read_text())
    sequences = meta["sequences"]
    counts: dict = {}
    with open(out / "per_circuit.csv") as handle:
        for row in csv.DictReader(handle):
            key = (float(row["p"]), int(row["depth"]), row["decoder"])
            counts.setdefault(key, {})[int(row["seq"])] = (
                int(row["errors"]), int(row["shots"]))
    rows = []
    for p in meta["p_list"]:
        for depth in meta["depths"]:
            row = {"p": p, "depth": depth}
            for name in DECODERS:
                per_seq = counts[(p, depth, name)]
                fid = np.array([abs(1 - 2 * e / n) for e, n in
                                (per_seq[s] for s in sorted(per_seq))])
                row["F_" + name] = fid.mean()
                row["F_" + name + "_sem"] = fid.std(ddof=1) / np.sqrt(len(fid))
            for scale, name in ((1.0, "F_rb_p"), (5.0, "F_rb_5p")):
                fid = np.array([physical_rb_fidelity(seq, depth, scale * p)
                                for seq in sequences])
                row[name] = fid.mean()
                row[name + "_seq_std"] = fid.std(ddof=1)
            rows.append(row)
    write_csv(out / "physical_rb_comparison.csv", rows)


def summarize(results: dict, p_list, depths, num_seq, out: Path) -> None:
    ler_rows, acc_rows, cal_rows, circ_rows = [], [], [], []
    hist_store = {}
    for pi, p in enumerate(p_list):
        for di, depth in enumerate(depths):
            parts = [results[(pi, si, di)] for si in range(num_seq)]
            shots = np.array([r["shots"] for r in parts], dtype=float)
            mld_err = np.array([r["errors"]["mld"] for r in parts], float)
            for si, r in enumerate(parts):
                for name in DECODERS:
                    circ_rows.append({
                        "p": p, "depth": depth, "seq": si, "decoder": name,
                        "shots": r["shots"], "errors": r["errors"][name],
                        "agree_mld": r["agree"][name],
                        "wrong_mldright": r["wrong_mldright"][name],
                        "right_mldwrong": r["right_mldwrong"][name]})
            for name in DECODERS:
                err = np.array([r["errors"][name] for r in parts], float)
                agree = np.array([r["agree"][name] for r in parts], float)
                n10 = np.array([r["wrong_mldright"][name] for r in parts],
                               float)
                n01 = np.array([r["right_mldwrong"][name] for r in parts],
                               float)
                v = np.stack([err, shots], axis=1)
                ler, ler_se = jackknife(v, lambda s: _ratio(s[0], s[1]))
                vr = np.stack([err, mld_err], axis=1)
                ratio, ratio_se = jackknife(vr, lambda s: _ratio(s[0], s[1]))
                va = np.stack([agree, shots], axis=1)
                agr, agr_se = jackknife(va, lambda s: _ratio(s[0], s[1]))
                vd = np.stack([err - mld_err, shots], axis=1)
                delta, delta_se = jackknife(vd, lambda s: _ratio(s[0], s[1]))
                total = shots.sum()
                row = {"p": p, "depth": depth, "decoder": name,
                       "shots": int(total), "errors": int(err.sum()),
                       "ler": ler, "ler_se_jk": ler_se,
                       "ler_se_binom": np.sqrt(max(ler * (1 - ler), 0) / total),
                       "fidelity": abs(1 - 2 * ler),
                       "ratio_to_mld": ratio, "ratio_se_jk": ratio_se,
                       "delta_vs_mld": delta, "delta_se_jk": delta_se,
                       "delta_se_mcnemar": np.sqrt(n10.sum() + n01.sum())
                       / total,
                       "agree_mld": agr, "agree_se_jk": agr_se,
                       "n_wrong_mldright": int(n10.sum()),
                       "n_right_mldwrong": int(n01.sum())}
                if name in SOFT:
                    ties = sum(r["ties"][name] for r in parts)
                    row.update(
                        tie_frac=ties / total,
                        tie_ler=_ratio(sum(r["tie_errors"][name]
                                           for r in parts), ties),
                        mld_ler_on_ties=_ratio(
                            sum(r["mld_errors_on_ties"][name]
                                for r in parts), ties))
                if name == "mwpm":
                    row["pm_argmin_disagree"] = sum(
                        r["pm_argmin_disagree"] for r in parts)
                    row["pm_weight_maxdiff"] = max(
                        r["pm_weight_maxdiff"] for r in parts)
                ler_rows.append(row)
            # Acceptance curves.
            zs = np.array([r["zero_syndrome"] for r in parts], float)
            a0, a0_se = jackknife(np.stack([zs[:, 0], shots], 1),
                                  lambda s: _ratio(s[0], s[1]))
            e0, e0_se = jackknife(zs, lambda s: _ratio(s[1], s[0]))
            acc_rows.append({"p": p, "depth": depth,
                             "decoder": "detect_all_zero_syndrome",
                             "target": "zero_syndrome", "acceptance": a0,
                             "acceptance_se_jk": a0_se,
                             "accepted_error": e0, "accepted_error_se_jk": e0_se,
                             "accepted_shots": zs[:, 0].sum(),
                             "accepted_wrong": zs[:, 1].sum()})
            targets = list(TARGET_ACCEPTANCE) + ["zero_syndrome"]
            for name in SOFT:
                hists = np.stack([r["hist"][name] for r in parts])
                hist_store[f"p{pi}_d{depth}_{name}"] = hists
                for target in targets:
                    level = a0 if target == "zero_syndrome" else target

                    def stat(h, level=level):
                        _, wrong, acc = error_at_acceptance(h, level)
                        return _ratio(wrong, acc)

                    flat = hists.reshape(num_seq, -1)
                    err_at, err_se = jackknife(
                        flat, lambda s, stat=stat: stat(s.reshape(-1, 2)))
                    _, wrong, acc = error_at_acceptance(hists.sum(0), level)
                    acc_rows.append({"p": p, "depth": depth, "decoder": name,
                                     "target": target, "acceptance": level,
                                     "accepted_error": err_at,
                                     "accepted_error_se_jk": err_se,
                                     "accepted_shots": acc,
                                     "accepted_wrong": wrong})
                cal = sum(r["calib"][name] for r in parts)
                for b in range(len(CALIB_EDGES) - 1):
                    if cal[b, 0] == 0:
                        continue
                    cal_rows.append({
                        "p": p, "depth": depth, "decoder": name,
                        "gap_lo": CALIB_EDGES[b], "gap_hi": CALIB_EDGES[b + 1],
                        "shots": int(cal[b, 0]), "errors": int(cal[b, 1]),
                        "empirical_error": cal[b, 1] / cal[b, 0],
                        "empirical_se": np.sqrt(cal[b, 1] / cal[b, 0]
                                                * (1 - cal[b, 1] / cal[b, 0])
                                                / cal[b, 0]),
                        "predicted_error_from_gap": cal[b, 2] / cal[b, 0]})
    write_csv(out / "ler.csv", ler_rows)
    write_csv(out / "acceptance.csv", acc_rows)
    write_csv(out / "calibration.csv", cal_rows)
    write_csv(out / "per_circuit.csv", circ_rows)
    np.savez_compressed(out / "gap_hist.npz", gap_edges=GAP_EDGES,
                        p_list=np.array(p_list), depths=np.array(depths),
                        **hist_store)


def _raw_sample_path(raw: Path, p: float, seq_index: int, depth: int) -> Path:
    return raw / f"sample_p{p:g}_seq{seq_index}_depth{depth}.pkl"


def _raw_structure_path(raw: Path, seq_index: int, depth: int) -> Path:
    return raw / f"structure_seq{seq_index}_depth{depth}.pkl"


def _load(path: Path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _dump(path: Path, value) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as handle:
        pickle.dump(value, handle)
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--raw-dir", default=None,
                        help="per-task pickles (default <out>/raw)")
    parser.add_argument("--p-list", default="0.0005,0.002,0.005,0.01")
    parser.add_argument("--depths", default="0,2,6,14,22")
    parser.add_argument("--num-seq", type=int, default=10)
    parser.add_argument("--shots", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seq-seed", type=int, default=20261028)
    parser.add_argument("--seed", type=int, default=2026_1028_2)
    parser.add_argument("--skip-structure", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--physical-rb-only", action="store_true",
                        help="only rebuild physical_rb_comparison.csv from "
                             "an existing summary in --out")
    args = parser.parse_args()
    if args.physical_rb_only:
        physical_rb_comparison(Path(args.out))
        return 0

    p_list = [float(x) for x in args.p_list.split(",")]
    depths = [int(x) for x in args.depths.split(",")]
    sequences = E.clifford_sequences(args.num_seq, 22, args.seq_seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    raw = Path(args.raw_dir) if args.raw_dir else out / "raw"
    raw.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    if not args.summarize_only:
        sample_tasks = []
        for p in p_list:
            for si in range(args.num_seq):
                for depth in depths:
                    seed = seed_for(args.seed, p, si, depth)
                    path = _raw_sample_path(raw, p, si, depth)
                    if path.exists():
                        old = _load(path)
                        if old["shots"] == args.shots and old["seed"] == seed:
                            continue
                    sample_tasks.append((0, p, si, sequences[si], 0, depth,
                                         args.shots, seed))
        # Longest tasks first for better load balance.
        sample_tasks.sort(key=lambda t: -(t[5] + 1) * (1 + 10 * t[1]))
        structure_tasks = [] if args.skip_structure else [
            (si, sequences[si], depth) for si in range(args.num_seq)
            for depth in depths
            if not _raw_structure_path(raw, si, depth).exists()]
        total = len(sample_tasks) + len(structure_tasks)
        print(f"{len(sample_tasks)} sample and {len(structure_tasks)} "
              f"structure tasks to run", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(sample_task, t): ("sample", t)
                       for t in sample_tasks}
            futures.update({pool.submit(structure_task, t): ("structure", t)
                            for t in structure_tasks})
            for count, future in enumerate(as_completed(futures), 1):
                kind, task = futures[future]
                value = future.result()
                if kind == "sample":
                    value["p"], value["seed"] = task[1], task[7]
                    value["seq"], value["depth"] = task[2], task[5]
                    _dump(_raw_sample_path(raw, task[1], task[2], task[5]),
                          value)
                else:
                    _dump(_raw_structure_path(raw, task[0], task[2]), value)
                if count % max(1, total // 25) == 0 or count == total:
                    print(f"[{time.perf_counter() - start:7.1f}s] {count}/"
                          f"{total} tasks", flush=True)

    results: dict = {}
    missing = []
    shots_used: dict = {}
    for pi, p in enumerate(p_list):
        for si in range(args.num_seq):
            for di, depth in enumerate(depths):
                path = _raw_sample_path(raw, p, si, depth)
                if not path.exists():
                    missing.append(path.name)
                    continue
                value = _load(path)
                value["key"] = (pi, si, di)
                results[(pi, si, di)] = value
                shots_used.setdefault(f"{p:g}", set()).add(value["shots"])
    if missing:
        print(f"missing {len(missing)} raw sample files, e.g. {missing[:3]}",
              flush=True)
        return 1
    census_rows, first_rows = [], []
    for si in range(args.num_seq):
        for depth in depths:
            path = _raw_structure_path(raw, si, depth)
            if path.exists():
                value = _load(path)
                census_rows.append(value["census"])
                first_rows.extend(value["first_order"])
    census_rows.sort(key=lambda r: (r["depth"], r["seq"]))
    first_rows.sort(key=lambda r: (r["p"], r["depth"], r["seq"]))
    write_csv(out / "census.csv", census_rows)
    write_csv(out / "first_order.csv", first_rows)
    summarize(results, p_list, depths, args.num_seq, out)
    meta = {"p_list": p_list, "depths": depths, "num_seq": args.num_seq,
            "shots_per_circuit": {k: sorted(v) for k, v in shots_used.items()},
            "seq_seed": args.seq_seed, "seed": args.seed,
            "seed_rule": "seed + 1_000_000*round(p*1e5) + 1000*seq + depth",
            "sequences": sequences, "decoders": DECODERS, "soft": SOFT,
            "first_order_p": FIRST_ORDER_P, "grid_p": GRID_P,
            "gap_bin_width": 0.02, "tie_tol": D.TIE_TOL,
            "sample_cpu_seconds": sum(r["seconds"] for r in results.values())}
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    physical_rb_comparison(out)
    print(f"summary written in {time.perf_counter() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
