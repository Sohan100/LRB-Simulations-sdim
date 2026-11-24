"""WP-E task 3: rule-based hybrids vs the gap rule on fresh, identical shots.

For every (p, sequence, depth) the script samples fresh shots from the
``data5`` circuit (seeds disjoint from the main grid), computes the exact LLR
with all detectors (``mld5``) and with the check rounds only (``mld2``), and
scores on the SAME shots:

* LRB-D ``const=1..22`` and ``unif=1..22`` (data-2 semantics) and every
  policy with the final data check (``*_final``);
* hybrid cluster rules from ``lrb_qubit.acceptance_rules``, each with the
  check rounds only (``_d2``) or with the final data detectors (``_d5``):

  ``tl``     isolated time-like detection pairs (flipped ancilla
             measurements) and lone terminal Z-check events; identity
             correction; decoder-free.
  ``tlq1``   ``tl`` plus single-round qubit-1 syndromes; identity correction;
             decoder-free.
  ``lut0``   single-fault lookup per cluster, only signatures whose DEM
             mechanisms agree on the logical flip; corrects that flip.
  ``lut10``  as ``lut0`` but accepts signatures with first-order posterior
             odds >= 10.

* the gap rule matched to every rule's accepted count on the same circuit
  (``MatchedGapScorer``): an exact paired Neyman-Pearson test.

Run (heavy part on the shared allocation)::

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python studies/correction-decoders/qubit/scripts/hybrid_rules.py run --p-indices 3,4,5,6,7,8 \
        --shots 200000 --workers 24 --raw <dir>
    python studies/correction-decoders/qubit/scripts/hybrid_rules.py analyse --raw <dir>
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import acceptance_rules as A  # noqa: E402
from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

DOCS = ROOT / "studies" / "correction-decoders" / "qubit"
PROBS = [0.0, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.0075, 0.01,
         0.0125, 0.015, 0.02, 0.03, 0.04, 0.05]
FINE_EDGES = np.concatenate([np.arange(0.0, 30.0, 0.05), [np.inf]])
SEED_BASE = 9_000_000_000
POLICIES = ([("const", k) for k in range(1, 23)]
            + [("unif", k) for k in range(1, 23)])
HYBRIDS = ("tl", "tlq1", "lut0", "lut10")
FIT_THRESHOLDS = (0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0)
FIT_LRBD = ("lrbd_unif_1", "lrbd_unif_2", "lrbd_unif_5", "lrbd_const_1",
            "lrbd_const_2", "lrbd_const_5", "lrbd_unif_1_final",
            "lrbd_const_1_final")

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"


def build_rules(l5, l2, dem5, dem2) -> dict[str, tuple]:
    w5 = A.dem_signature_weights(l5, dem5)
    w2 = A.dem_signature_weights(l2, dem2)
    rules = {}
    for mode, layout, dem, weights in (("d2", l2, dem2, w2),
                                       ("d5", l5, dem5, w5)):
        rules[f"tl_{mode}"] = (mode, layout, A.structural_tables(layout,
                                                                 False),
                               dem, weights)
        rules[f"tlq1_{mode}"] = (mode, layout,
                                 A.structural_tables(layout, True), dem,
                                 weights)
        rules[f"lut0_{mode}"] = (mode, layout, A.dem_lookup_tables(
            layout, dem, 0.0, weights), dem, weights)
        rules[f"lut10_{mode}"] = (mode, layout, A.dem_lookup_tables(
            layout, dem, 0.1, weights), dem, weights)
    return rules


def _fine_hist(llr: np.ndarray, flip: np.ndarray) -> np.ndarray:
    gap = np.abs(llr)
    wrong = (llr < 0) != flip
    bins = np.clip(np.searchsorted(FINE_EDGES, gap, side="right") - 1, 0,
                   len(FINE_EDGES) - 2)
    hist = np.zeros((len(FINE_EDGES) - 1, 2), dtype=np.int64)
    np.add.at(hist, (bins, wrong.astype(np.int64)), 1)
    return hist


def run_sequence(task) -> tuple[int, int, dict]:
    p_index, p, seq_index, sequence, depths, shots = task
    n_d = len(depths)
    out: dict[str, np.ndarray] = {}

    def put(name: str, d_index: int, value: float) -> None:
        if name not in out:
            out[name] = np.zeros(n_d)
        out[name][d_index] = value

    hist = {"mld2": np.zeros((n_d, len(FINE_EDGES) - 1, 2), np.int64),
            "mld5": np.zeros((n_d, len(FINE_EDGES) - 1, 2), np.int64)}
    for d_index, depth in enumerate(depths):
        lc5 = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
        lc2 = Q.build_lrb_circuit(sequence, depth, p, terminal="data2")
        dem5 = lc5.circuit.detector_error_model()
        dem2 = lc2.circuit.detector_error_model()
        mld5 = M.TransferMatrixMLD(dem5, M.lrb_detector_slots(lc5))
        mld2 = M.TransferMatrixMLD(dem2, M.lrb_detector_slots(lc2))
        l5 = A.SlotLayout.from_circuit(lc5, True)
        l2 = A.SlotLayout.from_circuit(lc5, False)
        rules = build_rules(l5, l2, dem5, dem2)
        sampler = lc5.circuit.compile_detector_sampler(
            seed=SEED_BASE + 1_000_000 * p_index + 1000 * seq_index
            + d_index)
        dets, obs = sampler.sample(shots, separate_observables=True)
        flip = obs[:, 0]
        n_round = 4 * lc5.num_rounds
        llr5 = mld5.log_likelihood_ratio(dets)
        llr2 = mld2.log_likelihood_ratio(dets[:, :n_round])
        hist["mld5"][d_index] = _fine_hist(llr5, flip)
        hist["mld2"][d_index] = _fine_hist(llr2, flip)
        scorers = {"mld2": A.MatchedGapScorer(llr2, flip),
                   "mld5": A.MatchedGapScorer(llr5, flip)}
        put("shots", d_index, shots)

        def record(name: str, accept: np.ndarray, rule_wrong: np.ndarray,
                   decoders: tuple[str, ...]) -> None:
            for dec in decoders:
                s = scorers[dec].score(accept, rule_wrong)
                put(f"{name}__accepted", d_index, s["accepted"])
                put(f"{name}__errors", d_index, s["rule_errors"])
                put(f"{name}__mlderr_{dec}", d_index,
                    s["mld_errors_on_accepted"])
                put(f"{name}__gaperr_{dec}", d_index, s["gap_errors"])
                put(f"{name}__diff_{dec}", d_index, s["diff"])
                put(f"{name}__diffvar_{dec}", d_index, s["diff_var"])

        raw = lc5.raw_syndromes(dets)
        final_clean = ~dets[:, n_round:].any(axis=1)
        for policy, k in POLICIES:
            acc = B.accept_mask(raw, lc5.rounds,
                                B.used_rounds(policy, k, depth))
            record(f"lrbd_{policy}_{k}", acc, flip, ("mld2", "mld5"))
            record(f"lrbd_{policy}_{k}_final", acc & final_clean, flip,
                   ("mld5",))
        codes = {"d2": l2.codes(dets[:, :n_round]), "d5": l5.codes(dets)}
        for name, (mode, layout, tables, dem, weights) in rules.items():
            acc, rule_flip = A.evaluate_cluster_rule(codes[mode], layout,
                                                     tables)
            decoders = ("mld2", "mld5") if mode == "d2" else ("mld5",)
            record(name, acc, rule_flip != flip, decoders)
            audit = A.single_fault_audit(layout, tables, dem, weights)
            for key in ("rejected", "accepted_ok", "accepted_wrong",
                        "undetectable_flip"):
                put(f"{name}__audit_{key}", d_index, audit[key])
    for dec, h in hist.items():
        out[f"hist_{dec}"] = h
    return p_index, seq_index, out


def run(args) -> int:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    raw_dir = Path(args.raw)
    raw_dir.mkdir(parents=True, exist_ok=True)
    meta = json.loads((Path(args.grid) / "meta.json").read_text())
    sequences = meta["sequences"][:args.num_seq]
    depths = [int(d) for d in args.depths.split(",")]
    p_indices = [int(i) for i in args.p_indices.split(",")]
    tasks = [(pi, PROBS[pi], si, sequences[si], depths, args.shots)
             for pi in p_indices for si in range(len(sequences))]
    parts: dict[int, dict[int, dict]] = {pi: {} for pi in p_indices}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_sequence, t) for t in tasks]
        for count, fut in enumerate(as_completed(futures), 1):
            pi, si, out = fut.result()
            parts[pi][si] = out
            if len(parts[pi]) == len(sequences):
                merged = {}
                for key in parts[pi][0]:
                    merged[key] = np.stack([parts[pi][s][key]
                                            for s in range(len(sequences))])
                merged["p"] = np.array(PROBS[pi])
                merged["depths"] = np.array(depths)
                merged["fine_edges"] = FINE_EDGES
                np.savez_compressed(raw_dir / f"hyb_p{pi}.npz", **merged)
                print(f"[{time.perf_counter() - start:7.1f}s] p{pi} saved",
                      flush=True)
            if count % max(1, len(tasks) // 10) == 0:
                print(f"[{time.perf_counter() - start:7.1f}s] {count}/"
                      f"{len(tasks)}", flush=True)
    (raw_dir / "hyb_meta.json").write_text(json.dumps({
        "p_indices": p_indices, "depths": depths, "shots": args.shots,
        "num_seq": len(sequences), "seed_base": SEED_BASE,
        "seq_seed": meta["seq_seed"]}, indent=1))
    return 0


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def rule_names(data: dict) -> list[str]:
    return sorted({k.split("__")[0] for k in data
                   if k.endswith("__accepted")})


def decoders_of(data: dict, name: str) -> list[str]:
    return [dec for dec in ("mld2", "mld5")
            if f"{name}__gaperr_{dec}" in data]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys = list(rows[0].keys())
    for row in rows[1:]:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v)
                             for k, v in row.items()})


def analyse_point(data: dict, idx: int, mult: np.ndarray
                  ) -> dict[str, list[dict]]:
    p = float(data["p"])
    depths = [int(d) for d in data["depths"]]
    shots = data["shots"]
    names = rule_names(data)
    cum = {}
    for dec in ("mld2", "mld5"):
        acc, err = A.cumulative_counts(data[f"hist_{dec}"])
        cum[dec] = (acc.astype(float), err.astype(float))
    ones = np.ones((1, mult.shape[1]), dtype=mult.dtype)
    weights = np.vstack([ones, mult])
    rows = {"points": [], "exact": [], "fits": [], "audit": []}
    for name in names:
        accepted = data[f"{name}__accepted"]
        errors = data[f"{name}__errors"]
        for dec in decoders_of(data, name):
            gaperr = data[f"{name}__gaperr_{dec}"]
            mlderr = data[f"{name}__mlderr_{dec}"]
            diff = data[f"{name}__diff_{dec}"]
            dvar = data[f"{name}__diffvar_{dec}"]
            acc_c, err_c = cum[dec]
            for d_index, depth in enumerate(depths):
                a = accepted[:, d_index].sum()
                n = shots[:, d_index].sum()
                e = errors[:, d_index].sum()
                g = gaperr[:, d_index].sum()
                dd = diff[:, d_index].sum()
                sd = np.sqrt(dvar[:, d_index].sum())
                rate = e / a if a > 0 else np.nan
                # acceptance gain at equal error (pooled fine-hist curve)
                pa = weights @ acc_c[:, d_index]
                pe = weights @ err_c[:, d_index]
                ba = weights @ accepted[:, d_index]
                be = weights @ errors[:, d_index]
                bn = weights @ shots[:, d_index]
                with np.errstate(invalid="ignore", divide="ignore"):
                    b_rate = be / ba
                _, _, gacc = A.batch_acceptance_at_error(
                    pa, pe, np.where(np.isfinite(b_rate), b_rate, 0.0))
                with np.errstate(invalid="ignore", divide="ignore"):
                    gain = (gacc / pa[:, 0]) / (ba / bn)
                good = gain[1:][np.isfinite(gain[1:])]
                rows["points"].append({
                    "p_idx": idx, "p": p, "rule": name, "decoder": dec,
                    "depth": depth, "shots": int(n), "acceptance": a / n,
                    "error_rate": rate,
                    "error_rate_se": np.sqrt(rate * (1 - rate) / a)
                    if a > 0 else np.nan,
                    "error_rate_mld_on_accepted": mlderr[:, d_index].sum() / a
                    if a > 0 else np.nan,
                    "gap_error_rate_same_acceptance": g / a
                    if a > 0 else np.nan,
                    "paired_diff_rate": dd / a if a > 0 else np.nan,
                    "paired_diff_se": sd / a if a > 0 else np.nan,
                    "paired_z": dd / sd if sd > 0 else np.nan,
                    "error_reduction": rate / (g / a)
                    if (a > 0 and g > 0) else np.nan,
                    "gap_acceptance_same_error": gacc[0] / pa[0, 0],
                    "acceptance_gain": gain[0],
                    "acceptance_gain_ci_low": float(np.quantile(good, 0.025))
                    if len(good) else np.nan,
                    "acceptance_gain_ci_high": float(np.quantile(good, 0.975))
                    if len(good) else np.nan})
        # fits (rule's own decision): hybrids and the key LRB-D policies
        if name.startswith("lrbd_") and name not in FIT_LRBD:
            continue
        fit = A.bootstrap_fixed_offset_fit(depths, accepted, errors, mult)
        mean, std = A.repo_fidelity_stats(accepted, errors)
        row = {"p_idx": idx, "p": p, "method": name, "f": fit["f"],
               "f_boot_std": fit["f_boot_std"], "f_ci_low": fit["f_ci_low"],
               "f_ci_high": fit["f_ci_high"], "b": fit["b"]}
        for d_index, depth in enumerate(depths):
            row[f"F_m{depth}"] = mean[d_index]
            row[f"Fstd_m{depth}"] = std[d_index]
            row[f"acc_m{depth}"] = (accepted[:, d_index].sum()
                                    / shots[:, d_index].sum())
        rows["fits"].append(row)
        if f"{name}__audit_rejected" in data:
            for d_index, depth in enumerate(depths):
                rows["audit"].append({
                    "p_idx": idx, "p": p, "rule": name, "depth": depth,
                    **{f"{key}_over_p": float(np.mean(
                        data[f"{name}__audit_{key}"][:, d_index]) / p)
                       for key in ("rejected", "accepted_ok",
                                   "accepted_wrong", "undetectable_flip")},
                    "first_order_acceptance": float(np.mean(np.exp(
                        -data[f"{name}__audit_rejected"][:, d_index])))})
    for dec in ("mld2", "mld5"):
        acc_c, err_c = cum[dec]
        for t in FIT_THRESHOLDS:
            k = int(np.flatnonzero(np.isclose(FINE_EDGES, t))[0])
            fit = A.bootstrap_fixed_offset_fit(depths, acc_c[:, :, k],
                                               err_c[:, :, k], mult)
            mean, std = A.repo_fidelity_stats(acc_c[:, :, k], err_c[:, :, k])
            row = {"p_idx": idx, "p": p, "method": f"gap_t{t:g}_{dec}",
                   "f": fit["f"], "f_boot_std": fit["f_boot_std"],
                   "f_ci_low": fit["f_ci_low"], "f_ci_high": fit["f_ci_high"],
                   "b": fit["b"]}
            for d_index, depth in enumerate(depths):
                row[f"F_m{depth}"] = mean[d_index]
                row[f"Fstd_m{depth}"] = std[d_index]
                row[f"acc_m{depth}"] = (acc_c[:, d_index, k].sum()
                                        / acc_c[:, d_index, 0].sum())
            rows["fits"].append(row)
    # Is a rule identical to the matched gap rule in every circuit?
    for name, dec in (("lrbd_unif_1", "mld2"), ("lrbd_unif_1_final", "mld5"),
                      ("lrbd_unif_1", "mld5")):
        if f"{name}__diff_{dec}" not in data:
            continue
        diff = data[f"{name}__diff_{dec}"]
        dvar = data[f"{name}__diffvar_{dec}"]
        acc = data[f"{name}__accepted"]
        used = acc > 0
        same = used & (np.abs(diff) < 1e-9) & (dvar < 1e-9)
        rows["exact"].append({
            "p_idx": idx, "p": p, "decoder": dec,
            "family": f"identity:{name}",
            "comparisons": int(used.sum()),
            "min_z": np.nan, "worst_rule": name, "worst_depth": "",
            "count_z_below_minus3": 0, "count_z_below_minus2": 0,
            "expected_below_minus2_if_null": np.nan,
            "median_z": np.nan,
            "circuits_identical_to_gap_rule": int(same.sum())})
    # exact NP summary on identical shots
    for dec in ("mld2", "mld5"):
        sel = [r for r in rows["points"] if r["decoder"] == dec
               and np.isfinite(r["paired_z"])]
        for family in ("lrbd", "hybrid"):
            fam = [r for r in sel if r["rule"].startswith("lrbd_")
                   == (family == "lrbd")]
            if not fam:
                continue
            z = np.array([r["paired_z"] for r in fam])
            worst = fam[int(np.argmin(z))]
            rows["exact"].append({
                "p_idx": idx, "p": p, "decoder": dec, "family": family,
                "comparisons": len(fam), "min_z": float(z.min()),
                "worst_rule": worst["rule"], "worst_depth": worst["depth"],
                "count_z_below_minus3": int(np.sum(z < -3)),
                "count_z_below_minus2": int(np.sum(z < -2)),
                "expected_below_minus2_if_null": 0.023 * len(fam),
                "median_z": float(np.median(z))})
    return rows


def _tidy_axis(ax, axis: str = "x") -> None:
    """Readable ticks: linear for spans under a decade, decade labels else."""
    from matplotlib import ticker
    lo, hi = ax.get_xlim() if axis == "x" else ax.get_ylim()
    target = ax.xaxis if axis == "x" else ax.yaxis
    if lo > 0 and hi / lo < 10.0:
        (ax.set_xscale if axis == "x" else ax.set_yscale)("linear")
        target.set_major_locator(ticker.MaxNLocator(4))
        target.set_major_formatter(ticker.FormatStrFormatter("%.3g"))
        target.set_minor_locator(ticker.NullLocator())
        return
    target.set_major_locator(ticker.LogLocator(base=10, numticks=7))
    target.set_major_formatter(ticker.LogFormatterSciNotation())
    target.set_minor_formatter(ticker.NullFormatter())


def plot(data_by_idx: dict, points: list[dict], fits: list[dict],
         figs: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def style(ax):
        ax.grid(True, color=GRID, linewidth=0.6)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=7)

    markers = {"tl": "^", "tlq1": "v", "lut0": "D", "lut10": "P",
               "lrbd_unif_1": "o", "lrbd_const_1": "s"}
    show_p = [i for i in data_by_idx if PROBS[i] in (0.001, 0.002, 0.005,
                                                      0.01)]
    show_d = (0, 6, 14, 22)
    fig, axes = plt.subplots(len(show_p), len(show_d),
                             figsize=(3.0 * len(show_d), 2.7 * len(show_p)),
                             squeeze=False)
    for row, idx in enumerate(show_p):
        data = data_by_idx[idx]
        depths = [int(d) for d in data["depths"]]
        for col, depth in enumerate(show_d):
            ax = axes[row, col]
            style(ax)
            if depth not in depths:
                ax.axis("off")
                continue
            d_index = depths.index(depth)
            for dec, color, ls in (("mld2", ORANGE, "--"),
                                   ("mld5", BLUE, "-")):
                acc, err = A.cumulative_counts(
                    data[f"hist_{dec}"][:, d_index].sum(axis=0))
                ok = (acc > 0) & (err > 0)  # zero-error points not drawn
                ax.plot(acc[ok] / acc[0], err[ok] / acc[ok],
                        color=color, ls=ls, lw=1.4)
            for r in points:
                if r["p_idx"] != idx or r["depth"] != depth:
                    continue
                base = r["rule"].rsplit("_", 1)[0] if not r["rule"].startswith(
                    "lrbd") else r["rule"].replace("_final", "")
                if base not in markers:
                    continue
                d5 = r["rule"].endswith("_d5") or r["rule"].endswith("_final")
                if r["decoder"] != ("mld5" if d5 else "mld2"):
                    continue
                if (not np.isfinite(r["error_rate"]) or r["acceptance"] <= 0
                        or r["error_rate"] <= 0):
                    continue
                color = BLUE if d5 else ORANGE
                ax.plot(r["acceptance"], r["error_rate"],
                        marker=markers[base], ms=5, ls="none", mec=color,
                        mfc=color if base.startswith(("lut", "tl"))
                        else "white", mew=1.0)
            ax.set_xscale("log")
            ax.set_yscale("log")
            _tidy_axis(ax, "x")
            _tidy_axis(ax, "y")
            ax.set_title(f"p = {PROBS[idx]:g}, m = {depth}", fontsize=8,
                         color=INK)
            if col == 0:
                ax.set_ylabel("error rate (accepted)", fontsize=7, color=INK2)
            if row == len(show_p) - 1:
                ax.set_xlabel("acceptance", fontsize=7, color=INK2)
    handles = [plt.Line2D([], [], color=ORANGE, ls="--",
                          label="gap rule, check rounds only"),
               plt.Line2D([], [], color=BLUE, ls="-",
                          label="gap rule, all detectors")]
    for base, mk in markers.items():
        handles.append(plt.Line2D([], [], color=INK2, marker=mk, ls="none",
                                  mfc=INK2 if base.startswith(("lut", "tl"))
                                  else "white",
                                  label=base.replace("lrbd_", "LRB-D ")))
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=7,
               frameon=False)
    fig.suptitle("Hybrid rules vs the gap rule on identical fresh shots "
                 "(orange: check rounds only; blue: with final data check)",
                 fontsize=9, color=INK)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(figs / "hybrids_pareto.png", dpi=130)
    plt.close(fig)

    # F(m) and acceptance(m)
    fig, axes = plt.subplots(2, len(show_p), figsize=(3.6 * len(show_p), 6),
                             squeeze=False)
    for col, idx in enumerate(show_p):
        ax_f, ax_a = axes[0, col], axes[1, col]
        style(ax_f)
        style(ax_a)
        depths = [int(d) for d in data_by_idx[idx]["depths"]]
        for method, color, marker in (
                ("lut0_d5", BLUE, "D"), ("tl_d5", BLUE, "^"),
                ("lut0_d2", ORANGE, "D"), ("tl_d2", ORANGE, "^"),
                ("lrbd_unif_1", INK2, "o"), ("lrbd_unif_1_final", INK2, "s"),
                ("gap_t4_mld5", AQUA, "x"), ("gap_t0_mld5", MUTED, "+")):
            sel = [r for r in fits if r["p_idx"] == idx
                   and r["method"] == method]
            if not sel:
                continue
            r = sel[0]
            ax_f.plot(depths, [r[f"F_m{d}"] for d in depths], color=color,
                      marker=marker, ms=4, lw=1.1,
                      mfc="white" if method.startswith("lrbd") else color,
                      label=method)
            acc_vals = np.array([r[f"acc_m{d}"] for d in depths], float)
            ax_a.plot(depths, np.where(acc_vals > 0, acc_vals, np.nan),
                      color=color, marker=marker, ms=4, lw=1.1,
                      mfc="white" if method.startswith("lrbd") else color)
        ax_f.set_title(f"p = {PROBS[idx]:g}", fontsize=9, color=INK)
        ax_f.set_ylim(0, 1.02)
        ax_a.set_yscale("log")
        ax_a.set_xlabel("depth m", fontsize=8, color=INK2)
        if col == 0:
            ax_f.set_ylabel("F (mean over sequences)", fontsize=8, color=INK2)
            ax_a.set_ylabel("acceptance", fontsize=8, color=INK2)
            ax_f.legend(fontsize=6, frameon=False)
    fig.tight_layout()
    fig.savefig(figs / "hybrids_F_and_acceptance.png", dpi=130)
    plt.close(fig)


def analyse(args) -> int:
    raw_dir = Path(args.raw)
    out = Path(args.out)
    figs = Path(args.figures)
    out.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    meta = json.loads((raw_dir / "hyb_meta.json").read_text())
    rng = np.random.default_rng(args.seed)
    mult = A.bootstrap_multiplicities(meta["num_seq"], args.n_boot, rng)
    collected = {"points": [], "exact": [], "fits": [], "audit": []}
    data_by_idx = {}
    for idx in meta["p_indices"]:
        path = raw_dir / f"hyb_p{idx}.npz"
        if not path.exists():
            continue
        data = dict(np.load(path))
        data_by_idx[idx] = data
        part = analyse_point(data, idx, mult)
        for key in collected:
            collected[key] += part[key]
        print(f"analysed p{idx}", flush=True)
    keep = ("lrbd_unif_1", "lrbd_unif_2", "lrbd_unif_5", "lrbd_const_1",
            "lrbd_const_2", "lrbd_const_5", "lrbd_unif_1_final",
            "lrbd_const_1_final")
    points = [r for r in collected["points"]
              if not r["rule"].startswith("lrbd_") or r["rule"] in keep]
    write_csv(out / "hybrid_points.csv", points)
    write_csv(out / "np_test_exact.csv", collected["exact"])
    write_csv(out / "np_test_exact_lrbd.csv", [
        {k: r[k] for k in ("p_idx", "p", "rule", "decoder", "depth",
                           "acceptance", "error_rate",
                           "gap_error_rate_same_acceptance",
                           "paired_diff_rate", "paired_diff_se", "paired_z")}
        for r in collected["points"] if r["rule"].startswith("lrbd_")])
    fits = [r for r in collected["fits"]
            if not r["method"].startswith("lrbd_") or r["method"] in keep]
    write_csv(out / "hybrid_fits.csv", fits)
    write_csv(out / "hybrid_audit.csv", collected["audit"])
    if not args.no_figures:
        plot(data_by_idx, collected["points"], collected["fits"], figs)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--grid", default=str(DOCS / "results" / "grid_main"))
    r.add_argument("--raw", required=True)
    r.add_argument("--p-indices", default="3,4,5,6,7,8")
    r.add_argument("--depths", default="0,2,4,6,10,14,18,20,22")
    r.add_argument("--num-seq", type=int, default=30)
    r.add_argument("--shots", type=int, default=200_000)
    r.add_argument("--workers", type=int, default=4)
    a = sub.add_parser("analyse")
    a.add_argument("--raw", required=True)
    a.add_argument("--out", default=str(DOCS / "results" / "E"))
    a.add_argument("--figures", default=str(DOCS / "figures" / "E"))
    a.add_argument("--n-boot", type=int, default=500)
    a.add_argument("--seed", type=int, default=777)
    a.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    return run(args) if args.cmd == "run" else analyse(args)


if __name__ == "__main__":
    raise SystemExit(main())
