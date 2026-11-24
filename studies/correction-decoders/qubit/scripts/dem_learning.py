"""Work package J, part 4: learning the detector error model from syndrome data.

On hardware the decoder prior (the DEM) must come from data. This module
learns it from detection-event correlations and measures what exact MLD loses
when it decodes with the learned model instead of the true one.

Estimators
----------
``pij_classic``
    The pairwise estimator of Chen et al. 2021 (Nature 595, 383; Supplement
    Eq. S11), p_ij = 1/2 - 1/2 sqrt(1 - 4 (<x_i x_j> - <x_i><x_j>) /
    (1 - 2<x_i> - 2<x_j> + 4<x_i x_j>)). Writing c_S = E[(-1)^{sum_{i in S} x_i}]
    it reads (1 - 2 p_ij)^2 = c_i c_j / c_ij, which for independent mechanisms
    equals the product of (1 - 2 p_e)^2 over ALL mechanisms e that flip both i
    and j (hyperedges included).
``estimate_footprints`` (hyperedge generalisation)
    For a set T of k detectors,
        sum_{e contains T} log(1 - 2 p_e) = (-1)^(k+1) 2^(1-k)
                                              sum_{S subset T} (-1)^(k-|S|) log c_S,
    exact for independent mechanisms (k = 2 is Eq. S11). Given a candidate
    set of footprints (the detector sets that the circuit structure allows),
    the probability of each exact footprint follows by subtracting candidate
    supersets, largest first (Moebius inversion on the superset lattice).

Learned DEM = learned footprint probabilities + an observable-split prior.
Which member of an ambiguous pair (e.g. Z0 vs Z2 at one location) fired is
invisible in the syndrome, so P(L flips | footprint) cannot be learned from
detection events at distance 2 (it is the d = 2 tie-break); it is taken from
a nominal model. The study separates the two error sources by also decoding
with learned rates + the true split ("oracle split").

CLI:
    python studies/correction-decoders/qubit/scripts/dem_learning.py study --workers 18
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from typing import Callable, Iterable, Sequence

if __name__ == "__main__":
    _THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    if any(os.environ.get(_v) != "1" for _v in _THREAD_VARS):
        for _v in _THREAD_VARS:
            os.environ[_v] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import stim

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
import hardware_model as H  # noqa: E402

from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

OUT = H.RESULTS / "dem_learning"
Footprint = tuple  # sorted tuple of detector indices

# ---------------------------------------------------------------------------
# DEM <-> footprint tables
# ---------------------------------------------------------------------------


def _xor_combine(p: float, q: float) -> float:
    return p + q - 2.0 * p * q


def footprint_table(dem: stim.DetectorErrorModel) -> dict:
    """footprint -> [p(L=0 part), p(L=1 part)] (XOR-combined mechanisms)."""
    table: dict = defaultdict(lambda: [0.0, 0.0])
    for mech in M.dem_mechanisms(dem):
        entry = table[tuple(mech.detectors)]
        k = int(mech.flips_observable)
        entry[k] = _xor_combine(entry[k], mech.probability)
    return dict(table)


def total_probability(entry: Sequence[float]) -> float:
    """Probability that the footprint fires (either observable class)."""
    return _xor_combine(entry[0], entry[1])


def split_ratio(entry: Sequence[float]) -> float:
    s = entry[0] + entry[1]
    return entry[1] / s if s > 0 else 0.0


def dem_from_table(table: dict, num_detectors: int) -> stim.DetectorErrorModel:
    lines = []
    for fp, (p0, p1) in sorted(table.items()):
        dets = " ".join(f"D{d}" for d in fp)
        if p0 > 0:
            lines.append(f"error({min(p0, 0.4999):.12g}) {dets}".strip())
        if p1 > 0:
            lines.append(f"error({min(p1, 0.4999):.12g}) {dets} L0".strip())
    lines.append(f"detector D{num_detectors - 1}")
    return stim.DetectorErrorModel("\n".join(lines))


def learned_table(probabilities: dict, split_from: dict) -> dict:
    """Learned footprint probabilities split by a prior table's L ratio."""
    out = {}
    for fp, p in probabilities.items():
        if p <= 0:
            continue
        rho = split_ratio(split_from[fp]) if fp in split_from else 0.0
        out[fp] = [p * (1.0 - rho), p * rho]
    # undetectable logical mechanisms are invisible to syndromes: keep prior
    for fp, entry in split_from.items():
        if len(fp) == 0:
            out[fp] = list(entry)
    return out


def scaled_table(table: dict, factor: float) -> dict:
    return {fp: [min(p0 * factor, 0.4999), min(p1 * factor, 0.4999)]
            for fp, (p0, p1) in table.items()}


def uniform_table(table: dict) -> dict:
    """Every mechanism class at the mean class probability (no calibration)."""
    values = [p for entry in table.values() for p in entry if p > 0]
    mean = float(np.mean(values))
    return {fp: [mean if p0 > 0 else 0.0, mean if p1 > 0 else 0.0]
            for fp, (p0, p1) in table.items()}


# ---------------------------------------------------------------------------
# Moments and estimators
# ---------------------------------------------------------------------------

_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


class ParityMoments:
    """c_S = E[(-1)^{x_S}] from detection events, with caching."""

    def __init__(self, dets: np.ndarray):
        dets = np.asarray(dets, dtype=bool)
        self.shots = dets.shape[0]
        self.packed = np.packbits(dets, axis=0)  # (ceil(N/8), D)
        self.cache: dict = {}

    def __call__(self, subset: Footprint) -> float:
        if not subset:
            return 1.0
        key = tuple(subset)
        if key not in self.cache:
            acc = self.packed[:, key[0]].copy()
            for d in key[1:]:
                acc ^= self.packed[:, d]
            odd = _POP8[acc].sum() / self.shots
            self.cache[key] = 1.0 - 2.0 * odd
        return self.cache[key]


def exact_moment_fn(table: dict) -> Callable[[Footprint], float]:
    """c_S computed exactly from a footprint table (for verification)."""
    entries = [(set(fp), total_probability(e)) for fp, e in table.items() if fp]

    def moment(subset):
        s = set(subset)
        value = 1.0
        for fp, p in entries:
            if len(fp & s) % 2:
                value *= 1.0 - 2.0 * p
        return value
    return moment


def _subsets(fp: Footprint):
    k = len(fp)
    for mask in range(1, 1 << k):
        yield tuple(fp[i] for i in range(k) if (mask >> i) & 1)


def superset_log_weight(moment: Callable, fp: Footprint) -> float:
    """sum over mechanisms e containing fp of log(1 - 2 p_e)."""
    k = len(fp)
    total = 0.0
    for sub in _subsets(fp):
        c = max(moment(sub), 1e-12)
        total += (-1) ** (k - len(sub)) * math.log(c)
    return (-1) ** (k + 1) * total / 2 ** (k - 1)


def estimate_footprints(moment: Callable, candidates: Iterable[Footprint]
                        ) -> dict:
    """Probabilities of the candidate footprints (generalised p_ij)."""
    cands = sorted({tuple(sorted(fp)) for fp in candidates if fp},
                   key=lambda fp: (-len(fp), fp))
    by_detector: dict = defaultdict(list)
    for fp in cands:
        for d in fp:
            by_detector[d].append(fp)
    weight: dict = {}
    for fp in cands:  # largest first, so supersets are done
        w = superset_log_weight(moment, fp)
        s = set(fp)
        for sup in by_detector[fp[0]]:
            if len(sup) > len(fp) and s.issubset(sup):
                w -= weight[sup]
        weight[fp] = min(w, 0.0)  # p >= 0
    return {fp: (1.0 - math.exp(w)) / 2.0 for fp, w in weight.items()}


def pij_classic(dets: np.ndarray, pairs: Iterable[tuple]) -> dict:
    """Chen et al. 2021 Eq. S11 for the given detector pairs."""
    x = np.asarray(dets, dtype=np.float64)
    mean = x.mean(axis=0)
    out = {}
    for i, j in pairs:
        xij = float(np.mean(x[:, i] * x[:, j]))
        num = xij - mean[i] * mean[j]
        den = 1 - 2 * mean[i] - 2 * mean[j] + 4 * xij
        arg = max(1 - 4 * num / den, 0.0)
        out[(i, j)] = 0.5 - 0.5 * math.sqrt(arg)
    return out


def pij_from_moments(moment: Callable, pairs: Iterable[tuple]) -> dict:
    """Eq. S11 written with parity moments: (1 - 2 p_ij)^2 = c_i c_j / c_ij."""
    out = {}
    for i, j in pairs:
        ratio = moment((i,)) * moment((j,)) / max(moment((i, j)), 1e-12)
        out[(i, j)] = 0.5 - 0.5 * math.sqrt(max(ratio, 0.0))
    return out


def true_pij(table: dict, pairs: Iterable[tuple]) -> dict:
    """XOR-combined probability of all mechanisms flipping both i and j."""
    out = {}
    for i, j in pairs:
        prod = 1.0
        for fp, entry in table.items():
            if i in fp and j in fp:
                prod *= 1.0 - 2.0 * total_probability(entry)
        out[(i, j)] = (1.0 - prod) / 2.0
    return out


# ---------------------------------------------------------------------------
# Decoding comparison
# ---------------------------------------------------------------------------


def decode_errors(dem: stim.DetectorErrorModel, slots, dets: np.ndarray,
                  flips: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    llr = M.TransferMatrixMLD(dem, slots).log_likelihood_ratio(dets)
    return (llr < 0) != flips, np.abs(llr)


def postselected_error(wrong: np.ndarray, gap: np.ndarray,
                       acceptance: float) -> float:
    """Logical error among the ``acceptance`` fraction of shots with the
    largest |LLR| (ties broken randomly via a stable order)."""
    n = len(gap)
    keep = max(int(round(acceptance * n)), 1)
    order = np.argsort(-gap, kind="stable")[:keep]
    return float(wrong[order].mean())


def _nominal_split_device() -> H.Device:
    return H.IQM_GARNET


def si1000_shaped(device: H.Device) -> H.Device:
    """The same circuit with SI1000 proportions at the device's p (CZ)."""
    p = H.depolarizing_from_infidelity(device.r_2q, 2)
    return replace(device, name=f"si1000_shape_{device.name}",
                   t1_us=math.inf, t2_us=math.inf, overrides=(),
                   r_1q=0.1 * p * 2 / 3, e_meas=5 * p, e_reset=2 * p,
                   extra_idle=(("meas", 2 * p), ("reset", 2 * p),
                               ("1q", 0.1 * p), ("2q", 0.1 * p)))


def study_circuit(args) -> dict:
    """Learn the DEM of one circuit from training shots; decode test shots."""
    (truth, nominal, sequence, depth, seq_index, n_train_list, n_test,
     seed, scales) = args
    t0 = time.perf_counter()
    lc = H.build_hw_lrb_circuit(sequence, depth, truth)
    slots = M.lrb_detector_slots(lc)
    true_dem = lc.circuit.detector_error_model()
    true_tab = footprint_table(true_dem)
    nom_lc = H.build_hw_lrb_circuit(sequence, depth, nominal)
    nom_tab = footprint_table(nom_lc.circuit.detector_error_model())
    shape_lc = H.build_hw_lrb_circuit(sequence, depth, si1000_shaped(nominal))
    shape_dem = shape_lc.circuit.detector_error_model()
    n_det = true_dem.num_detectors
    candidates = [fp for fp in nom_tab if fp]
    missing = [fp for fp in true_tab if fp and fp not in nom_tab]

    sampler = lc.circuit.compile_detector_sampler(seed=seed)
    test_dets, test_obs = sampler.sample(n_test, separate_observables=True)
    flips = test_obs[:, 0]

    decoders: dict = {"oracle": true_dem, "nominal": dem_from_table(
        nom_tab, n_det), "si1000_shape": shape_dem,
        "uniform": dem_from_table(uniform_table(true_tab), n_det)}
    for s in scales:
        decoders[f"scaled_{s:g}"] = dem_from_table(scaled_table(true_tab, s),
                                                   n_det)
    recovery = {}
    train_sampler = lc.circuit.compile_detector_sampler(seed=seed + 1)
    train_all, _ = train_sampler.sample(max(n_train_list),
                                        separate_observables=True)
    pairs = sorted({(fp[a], fp[b]) for fp in candidates
                    for a in range(len(fp)) for b in range(a + 1, len(fp))})
    tp = true_pij(true_tab, pairs)
    for n_train in n_train_list:
        train = train_all[:n_train]
        moments = ParityMoments(train)
        est = estimate_footprints(moments, candidates)
        graph = estimate_footprints(moments, [fp for fp in candidates
                                              if len(fp) <= 2])
        decoders[f"learned_{n_train}"] = dem_from_table(
            learned_table(est, nom_tab), n_det)
        decoders[f"learned_{n_train}_oracle_split"] = dem_from_table(
            learned_table(est, true_tab), n_det)
        decoders[f"graph_{n_train}"] = dem_from_table(
            learned_table(graph, nom_tab), n_det)
        # recovery accuracy of footprint probabilities and of p_ij
        truth_p = np.array([total_probability(true_tab.get(fp, (0, 0)))
                            for fp in est])
        est_p = np.array([est[fp] for fp in est])
        # batch standard errors for p_ij (10 batches)
        pc = pij_from_moments(moments, pairs)
        batches = np.array_split(np.arange(n_train), 10)
        pb = np.array([[v for v in pij_from_moments(
            ParityMoments(train[b]), pairs).values()] for b in batches])
        se = pb.std(axis=0, ddof=1) / math.sqrt(len(batches))
        diff = np.array([pc[k] - tp[k] for k in pairs])
        big = np.array([tp[k] > 1e-3 for k in pairs])
        recovery[n_train] = {
            "footprints": len(est),
            "abs_error_sum_over_truth_sum": float(
                np.abs(est_p - truth_p).sum() / truth_p.sum()),
            "pij_pairs": len(pairs),
            "pij_median_rel_err_ptrue_gt_1e-3": float(np.median(
                np.abs(diff[big]) / np.array([tp[k] for k in pairs])[big]))
            if big.any() else None,
            "pij_frac_within_3se": float(np.mean(np.abs(diff) <= 3 * se
                                                 + 1e-12)),
        }

    results = {}
    oracle_wrong, oracle_gap = decode_errors(decoders["oracle"], slots,
                                             test_dets, flips)
    for name, dem in decoders.items():
        wrong, gap = ((oracle_wrong, oracle_gap) if name == "oracle"
                      else decode_errors(dem, slots, test_dets, flips))
        d = wrong.astype(float) - oracle_wrong.astype(float)
        results[name] = {
            "errors": int(wrong.sum()), "shots": int(len(wrong)),
            "diff_vs_oracle": float(d.mean()),
            "diff_se": float(d.std(ddof=1) / math.sqrt(len(d))),
            "post_err_a0.5": postselected_error(wrong, gap, 0.5),
            "post_err_a0.8": postselected_error(wrong, gap, 0.8),
            "mean_predicted_error": float(np.mean(1.0 / (1.0 + np.exp(gap)))),
        }
    return {"seq": seq_index, "depth": depth, "detectors": n_det,
            "candidates": len(candidates), "missing_footprints": len(missing),
            "max_footprint": max(len(fp) for fp in true_tab),
            "recovery": recovery, "decoders": results,
            "seconds": time.perf_counter() - t0}


def aggregate(rows: list[dict]) -> dict:
    """Mean over circuits per depth, with SEM over circuits."""
    out: dict = {}
    depths = sorted({r["depth"] for r in rows})
    for depth in depths:
        sub = [r for r in rows if r["depth"] == depth]
        names = sub[0]["decoders"].keys()
        entry = {}
        eps_oracle = float(np.mean([r["decoders"]["oracle"]["errors"]
                                    / r["decoders"]["oracle"]["shots"]
                                    for r in sub]))
        for name in names:
            eps = np.array([r["decoders"][name]["errors"]
                            / r["decoders"][name]["shots"] for r in sub])
            diff = np.array([r["decoders"][name]["diff_vs_oracle"]
                             for r in sub])
            dse = np.array([r["decoders"][name]["diff_se"] for r in sub])
            p5 = np.array([r["decoders"][name]["post_err_a0.5"] for r in sub])
            p8 = np.array([r["decoders"][name]["post_err_a0.8"] for r in sub])
            pred = np.array([r["decoders"][name]["mean_predicted_error"]
                             for r in sub])
            n = len(sub)
            entry[name] = {
                "eps_mean": float(eps.mean()),
                "eps_sem": float(eps.std(ddof=1) / math.sqrt(n)) if n > 1
                else None,
                "diff_mean": float(diff.mean()),
                # paired: combine within-circuit shot noise of each diff
                "diff_se": float(math.sqrt((dse ** 2).sum()) / n),
                # relative excess error (eps - eps_oracle) / eps_oracle
                "rel_diff": float(diff.mean() / eps_oracle) if eps_oracle > 0
                else None,
                "rel_diff_se": float(math.sqrt((dse ** 2).sum()) / n
                                     / eps_oracle) if eps_oracle > 0
                else None,
                "post_err_a0.5": float(p5.mean()),
                "post_err_a0.8": float(p8.mean()),
                "mean_predicted_error": float(pred.mean()),
            }
        rec = {}
        for n_train in sub[0]["recovery"]:
            vals = [r["recovery"][n_train] for r in sub]
            rec[str(n_train)] = {
                k: (float(np.mean([v[k] for v in vals]))
                    if vals[0][k] is not None else None)
                for k in vals[0]}
        out[str(depth)] = {"circuits": len(sub), "decoders": entry,
                           "recovery": rec,
                           "detectors": float(np.mean([r["detectors"]
                                                       for r in sub])),
                           "max_footprint": int(max(r["max_footprint"]
                                                    for r in sub)),
                           "missing_footprints": int(sum(
                               r["missing_footprints"] for r in sub))}
    return out


def run_study(truth: H.Device, nominal: H.Device, tag: str, num_seq: int,
              depths: Sequence[int], n_train_list: Sequence[int],
              n_test: int, workers: int, seed: int = 777,
              scales=(0.1, 0.25, 0.5, 2.0, 4.0, 10.0)) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    sequences = E.clifford_sequences(num_seq, max(depths), H.SEQ_SEED)
    tasks = [(truth, nominal, sequences[si], depth, si, list(n_train_list),
              n_test, seed + 1000 * si + depth, scales)
             for si in range(num_seq) for depth in depths]
    rows = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(study_circuit, t) for t in tasks]
        for count, fut in enumerate(as_completed(futures), 1):
            rows.append(fut.result())
            print(f"[{time.perf_counter() - start:7.1f}s] {tag} {count}/"
                  f"{len(tasks)}", flush=True)
    summary = {"tag": tag, "truth": truth.name, "nominal": nominal.name,
               "num_seq": num_seq, "depths": list(depths),
               "n_train": list(n_train_list), "n_test": n_test,
               "per_depth": aggregate(rows), "rows": rows}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(summary, indent=1,
                                                default=str))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="WP J DEM learning study")
    sub = parser.add_subparsers(dest="cmd", required=True)
    st = sub.add_parser("study")
    st.add_argument("--truth", default="iqm_garnet")
    st.add_argument("--hetero-sigma", type=float, default=0.0)
    st.add_argument("--hetero-seed", type=int, default=11)
    st.add_argument("--tag", default=None)
    st.add_argument("--num-seq", type=int, default=6)
    st.add_argument("--depths", default="2,6,10")
    st.add_argument("--n-train", default="10000,100000,1000000")
    st.add_argument("--n-test", type=int, default=200000)
    st.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    nominal = H.DEVICES[args.truth]
    truth = (H.heterogeneous(nominal, args.hetero_sigma, args.hetero_seed)
             if args.hetero_sigma > 0 else nominal)
    tag = args.tag or (f"{args.truth}_het{args.hetero_sigma:g}"
                       if args.hetero_sigma > 0 else args.truth)
    summary = run_study(truth, nominal, tag, args.num_seq,
                        [int(d) for d in args.depths.split(",")],
                        [int(n) for n in args.n_train.split(",")],
                        args.n_test, args.workers)
    for depth, entry in summary["per_depth"].items():
        print(f"depth {depth}:")
        for name, v in entry["decoders"].items():
            print(f"  {name:32s} eps={v['eps_mean']:.4f} "
                  f"diff={v['diff_mean']:+.5f}+-{v['diff_se']:.5f} "
                  f"post50={v['post_err_a0.5']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
