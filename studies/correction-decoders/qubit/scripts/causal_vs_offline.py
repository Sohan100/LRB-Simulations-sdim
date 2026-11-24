"""Causal (real-time) versus offline decoding for qubit [[5,1,2]] LRB (WP D).

Subcommands
-----------
``grid``
    Sample the ``data5`` LRB circuits of 30 Clifford sequences (seed
    20261028) at depths 0..22 and decode the *same* shots with the offline
    exact MLD (``mld.TransferMatrixMLD``) and with the causal decoders of
    ``lrb_qubit.decoders_causal`` (state commit, lookahead 0, 1, 2, 4 and
    unbounded; logical commit, lookahead 0, 1 and unbounded). Writes
    ``<out>/raw/grid_p<p>.npz`` (counts only).
``report``
    Logical error rates, F(m) = |1 - 2e| averaged over sequences, repo-style
    fixed-offset fits F = b f^m with sequence-bootstrap errors, paired
    differences against offline MLD, local decay rates. Writes small CSVs.
``frame``
    Pauli-frame versus physical mid-circuit correction: exact DEM comparison
    of rule A built with Stim classically controlled Paulis, identical-noise
    FlipSimulator runs of rules A, B1, B2 (shot-by-shot comparison), and the
    effect of noisy correction gates.
``toys``
    Exact toy models where frame and physical correction differ
    (non-Clifford gate, leakage).
``perslot``
    Ground truth from DEM error sampling: how often the logical frame
    ``l_t`` and syndrome attribution ``b_t`` at each round are estimated
    wrongly by offline smoothing, by uncommitted fixed-lag estimates and by
    the committed causal decoders.

Example (shared allocation; see results/ALLOCATION.md). Per the lead's note,
use ``-c 256 --cpu-bind=none`` so overlapping steps are not all pinned to the
same few CPUs, and keep ``--workers`` at 24 or less:

    srun --jobid=<JOBID> --overlap --cpu-bind=none -N 1 -n 1 -c 256 bash -c \
"module load python/3.11; cd <WORKTREE>; export OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1; python studies/correction-decoders/qubit/scripts/causal_vs_offline.py grid \
--p 0.002:10000 --workers 24 --resume"
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

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import decoders_causal as D  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

DEFAULT_OUT = ROOT / "studies/correction-decoders/qubit/results/D"
SEQ_SEED = 20261028
REPO_DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]

# (name, commit, lookahead); "offline" is the lead's exact MLD with the
# explicit l0 tie rule (decoders_causal.offline_prediction).
CAUSAL = [("state_w0", "state", 0), ("state_w1", "state", 1),
          ("state_w2", "state", 2), ("state_w4", "state", 4),
          ("state_winf", "state", None),
          ("logical_w0", "logical", 0), ("logical_w1", "logical", 1),
          ("logical_winf", "logical", None)]
# Grid decoders: every state-commit decoder under every tie rule (exact
# posterior ties occur only in state mode); logical commit and offline never
# tie (checked: 0 tied decisions), so they run with the l0 rule only.
GRID_CAUSAL = ([(f"{n}_{r}", c, w, r) for n, c, w in CAUSAL if c == "state"
                for r in D.TIE_RULES]
               + [(f"{n}_l0", c, w, "l0") for n, c, w in CAUSAL
                  if c == "logical"])
DECODERS = ["offline"] + [name for name, _, _, _ in GRID_CAUSAL]


# ---------------------------------------------------------------------------
# grid
# ---------------------------------------------------------------------------

def _grid_task(args):
    p_tag, p, seq_index, sequence, depths, shots, seed, terminal = args
    n_dec = len(DECODERS)
    out = {key: np.zeros((n_dec, len(depths)), dtype=np.int64)
           for key in ("shots", "errors", "n10", "n01", "ties", "commits")}
    seconds = np.zeros(n_dec)
    start = time.perf_counter()
    for d_index, depth in enumerate(depths):
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal=terminal)
        dem = lc.circuit.detector_error_model()
        base = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
        sampler = lc.circuit.compile_detector_sampler(
            seed=seed + 7_919 * seq_index + 104_729 * d_index)
        dets, obs = sampler.sample(shots, separate_observables=True)
        flip = obs[:, 0]
        t0 = time.perf_counter()
        llr = base.log_likelihood_ratio(dets)
        offline = D.offline_prediction(llr)
        seconds[0] += time.perf_counter() - t0
        out["ties"][0, d_index] = int((np.abs(llr) <= D.TIE_TOL).sum())
        out["commits"][0, d_index] = shots
        wrong_off = offline != flip
        preds = [offline]
        for name, commit, w, rule in GRID_CAUSAL:
            k = DECODERS.index(name)
            dec = D.CausalWindowMLD(None, None, w, commit, base=base,
                                    tie_rule=rule,
                                    seed=seed + 31 * seq_index + d_index)
            t0 = time.perf_counter()
            preds.append(dec.decode(dets))
            seconds[k] += time.perf_counter() - t0
            out["ties"][k, d_index] = dec.tie_commits
            out["commits"][k, d_index] = dec.total_commits
            if dec.dead_shots:
                raise RuntimeError(f"{name}: {dec.dead_shots} dead shots")
        for k, pred in enumerate(preds):
            wrong = pred != flip
            out["shots"][k, d_index] = shots
            out["errors"][k, d_index] = int(wrong.sum())
            out["n10"][k, d_index] = int((wrong & ~wrong_off).sum())
            out["n01"][k, d_index] = int((~wrong & wrong_off).sum())
    return seq_index, out, seconds, time.perf_counter() - start


def cmd_grid(args) -> int:
    """All (p, sequence) tasks share one process pool; each p has its own
    incrementally saved npz (``done`` mask), so ``--resume`` reruns only the
    missing sequences."""
    depths = list(range(0, 23)) if args.depths == "all" else [
        int(d) for d in args.depths.split(",")]
    sequences = E.clifford_sequences(args.num_seq, 22, SEQ_SEED)
    out = Path(args.out) / "raw"
    out.mkdir(parents=True, exist_ok=True)
    n_dec = len(DECODERS)
    states = {}
    tasks = []
    for item in args.p.split(","):
        p_text, _, shots_text = item.partition(":")
        p = float(p_text)
        shots = int(shots_text) if shots_text else args.shots
        p_tag = f"{p:g}"
        path = out / f"grid_{args.terminal}_p{p_tag}.npz"
        arrays = {key: np.zeros((n_dec, args.num_seq, len(depths)),
                                dtype=np.int64)
                  for key in ("shots", "errors", "n10", "n01", "ties",
                              "commits")}
        done = np.zeros(args.num_seq, dtype=bool)
        seconds = np.zeros(n_dec)
        if args.resume and path.exists():
            old_meta, old = _load(path)
            if (old_meta["depths"] == depths and old_meta["shots"] == shots
                    and old_meta["decoders"] == DECODERS):
                done = np.asarray(old_meta.get("done", [True] *
                                               args.num_seq), dtype=bool)
                for key in arrays:
                    arrays[key][:, done, :] = old[key][:, done, :]
                seconds = np.asarray(old_meta["decode_seconds"])
        states[p_tag] = dict(p=p, shots=shots, path=path, arrays=arrays,
                             done=done, seconds=seconds)
        tasks += [(p_tag, p, i, sequences[i], depths, shots,
                   args.seed + int(round(p * 1e7)), args.terminal)
                  for i in range(args.num_seq) if not done[i]]
    start = time.perf_counter()

    def save(st):
        meta = {"p": st["p"], "depths": depths, "num_seq": args.num_seq,
                "seq_seed": SEQ_SEED, "shots": st["shots"],
                "seed": args.seed, "terminal": args.terminal,
                "decoders": DECODERS,
                "causal": [[n, c, w, r] for n, c, w, r in GRID_CAUSAL],
                "tie_tol": D.TIE_TOL,
                "decode_seconds": st["seconds"].tolist(),
                "done": st["done"].tolist(),
                "wall_seconds": time.perf_counter() - start}
        tmp = st["path"].with_suffix(".tmp.npz")
        np.savez_compressed(tmp, meta=json.dumps(meta), **st["arrays"])
        os.replace(tmp, st["path"])

    print(f"{len(tasks)} tasks to run", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_grid_task, t): t[0] for t in tasks}
        for count, fut in enumerate(as_completed(futures), 1):
            p_tag = futures[fut]
            st = states[p_tag]
            seq_index, part, sec, wall = fut.result()
            for key in st["arrays"]:
                st["arrays"][key][:, seq_index, :] = part[key]
            st["seconds"] += sec
            st["done"][seq_index] = True
            save(st)
            print(f"[{time.perf_counter() - start:7.1f}s] p={p_tag} "
                  f"seq {seq_index} done ({wall:.1f}s) {count}/"
                  f"{len(tasks)}", flush=True)
    for p_tag, st in states.items():
        save(st)
        print(f"wrote {st['path']} ({int(st['done'].sum())}/{args.num_seq} "
              "sequences)", flush=True)
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _load(path):
    data = np.load(path)
    meta = json.loads(str(data["meta"]))
    keys = [k for k in ("shots", "errors", "n10", "n01", "ties", "commits")
            if k in data.files]
    return meta, {k: data[k] for k in keys}


def _fid(errors, shots):
    return np.abs(1.0 - 2.0 * errors / shots)


def _fit(depths, fid_seq, depth_mask=None):
    """Repo fit on the sequence mean, weights 1/std^2 (std over sequences)."""
    depths = np.asarray(depths)
    mask = np.ones(len(depths), bool) if depth_mask is None else depth_mask
    mean = fid_seq.mean(axis=0)[mask]
    std = fid_seq.std(axis=0)[mask]
    return D.fit_fixed_offset(depths[mask], mean, std)


def _proper_chi2(depths, fid_seq, fit, depth_mask):
    depths = np.asarray(depths)[depth_mask]
    mean = fid_seq.mean(axis=0)[depth_mask]
    se = fid_seq.std(axis=0, ddof=1)[depth_mask] / np.sqrt(fid_seq.shape[0])
    model = fit["b"] * fit["f"] ** depths
    return float(np.sum(((mean - model) / se) ** 2)), int(len(depths) - 2)


def _window_rate(fk, idx, depths, lo, hi):
    i = int(np.flatnonzero(depths == lo)[0])
    j = int(np.flatnonzero(depths == hi)[0])
    return -np.log(fk[idx, j].mean() / fk[idx, i].mean()) / (hi - lo)


def _write_csv(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v)
                             for k, v in row.items()})
    print(f"wrote {path} ({len(rows)} rows)")


def cmd_report(args) -> int:
    """Main tables use the explicit l0 tie rule; the tie-rule spread of the
    state-commit decoders is reported separately as a systematic."""
    root = Path(args.out)
    files = sorted((root / "raw").glob(f"grid_{args.terminal}_p*.npz"))
    if not files:
        print("no grid files")
        return 1
    rng = np.random.default_rng(12345)
    summary_rows, fit_rows, paired_rows, rate_rows, tie_rows = \
        [], [], [], [], []
    for path in files:
        meta, a = _load(path)
        done = np.asarray(meta.get("done", [True] * meta["num_seq"]), bool)
        a = {k: v[:, done, :] for k, v in a.items()}
        p = meta["p"]
        depths = np.asarray(meta["depths"])
        names = meta["decoders"]
        n_seq = int(done.sum())
        main = [n for n in names if n == "offline" or n.endswith("_l0")]
        fid = _fid(a["errors"], a["shots"])           # (dec, seq, depth)
        err = a["errors"] / a["shots"]
        boot_idx = rng.integers(0, n_seq, size=(args.boot, n_seq))
        masks = {"all": np.ones(len(depths), bool),
                 "repo9": np.isin(depths, REPO_DEPTHS)}
        off = names.index("offline")
        fits = {}
        for k, name in enumerate(names):
            for mask_name, mask in masks.items():
                if name not in main and mask_name != "all":
                    continue
                fit = _fit(depths, fid[k], mask)
                chi2, dof = _proper_chi2(depths, fid[k], fit, mask)
                boots = np.asarray([[bf["f"], bf["b"]] for bf in
                                    (_fit(depths, fid[k][idx], mask)
                                     for idx in boot_idx)])
                fits[(name, mask_name)] = (fit, boots)
                row = {"p": p, "decoder": name, "depths": mask_name,
                       "f": fit["f"], "f_se_boot": boots[:, 0].std(ddof=1),
                       "b": fit["b"], "b_se_boot": boots[:, 1].std(ddof=1),
                       "r_avg_infidelity": (1 - fit["f"]) / 2,
                       "minus_ln_f": -np.log(fit["f"]),
                       "chi2_se_weighted": chi2, "dof": dof,
                       "tie_fraction": a["ties"][k].sum()
                       / max(a["commits"][k].sum(), 1)}
                if name != "offline":
                    fo, bo = fits[("offline", mask_name)]
                    ratio = (1 - boots[:, 0]) / (1 - bo[:, 0])
                    row["rate_ratio_vs_offline"] = ((1 - fit["f"])
                                                    / (1 - fo["f"]))
                    row["rate_ratio_se_paired_boot"] = ratio.std(ddof=1)
                fit_rows.append(row)
            if name not in main:
                continue
            for d_index, depth in enumerate(depths):
                e = err[k, :, d_index]
                f = fid[k, :, d_index]
                summary_rows.append({
                    "p": p, "decoder": name, "depth": int(depth),
                    "shots_per_seq": int(a["shots"][k, 0, d_index]),
                    "logical_error_mean": e.mean(),
                    "logical_error_se": e.std(ddof=1) / np.sqrt(n_seq),
                    "F_mean": f.mean(), "F_std": f.std(),
                    "F_se": f.std(ddof=1) / np.sqrt(n_seq)})
            if name == "offline":
                continue
            for d_index, depth in enumerate(depths):
                diff = err[k, :, d_index] - err[off, :, d_index]
                n10 = int(a["n10"][k, :, d_index].sum())
                n01 = int(a["n01"][k, :, d_index].sum())
                total = int(a["shots"][k, :, d_index].sum())
                paired_rows.append({
                    "p": p, "decoder": name, "depth": int(depth),
                    "delta_error_mcnemar": (n10 - n01) / total,
                    "delta_error_mcnemar_se": np.sqrt(n10 + n01) / total,
                    "delta_error_se_seq": diff.std(ddof=1) / np.sqrt(n_seq),
                    "n10_causal_wrong_offline_right": n10,
                    "n01_offline_wrong_causal_right": n01})
        # Tie-rule spread of the state-commit decoders (systematic).
        for base_name, commit, w in CAUSAL:
            if commit != "state":
                continue
            ref = f"{base_name}_l0"
            fit_ref, boots_ref = fits[(ref, "all")]
            for rule in D.TIE_RULES:
                name = f"{base_name}_{rule}"
                k = names.index(name)
                fit_k, boots_k = fits[(name, "all")]
                fo, bo = fits[("offline", "all")]
                ratio_b = (1 - boots_k[:, 0]) / (1 - bo[:, 0])
                ratio_ref_b = (1 - boots_ref[:, 0]) / (1 - bo[:, 0])
                row = {"p": p, "decoder": base_name, "tie_rule": rule,
                       "rate_ratio_vs_offline": (1 - fit_k["f"])
                       / (1 - fo["f"]),
                       "rate_ratio_se_paired_boot": ratio_b.std(ddof=1),
                       "shift_vs_l0": (1 - fit_k["f"]) / (1 - fo["f"])
                       - (1 - fit_ref["f"]) / (1 - fo["f"]),
                       "shift_vs_l0_se_paired_boot":
                           (ratio_b - ratio_ref_b).std(ddof=1),
                       "tie_fraction_of_commits": a["ties"][k].sum()
                       / max(a["commits"][k].sum(), 1)}
                for m in (1, 2, 4, 10):
                    if m not in depths:
                        continue
                    d_index = int(np.flatnonzero(depths == m)[0])
                    n10 = int(a["n10"][k, :, d_index].sum())
                    n01 = int(a["n01"][k, :, d_index].sum())
                    total = int(a["shots"][k, :, d_index].sum())
                    row[f"excess_error_m{m}"] = (n10 - n01) / total
                    row[f"excess_error_m{m}_se"] = np.sqrt(n10 + n01) / total
                tie_rows.append(row)
        # Local rates and a curvature index, on windows where F stays well
        # above its noise floor: (2-10, 14-22) at p <= 2e-3 and (2-8, 8-14)
        # at p = 5e-3 (F(22) ~ 0.03 there).
        windows = [(0, 2), (2, 6), (6, 10), (10, 14), (14, 18), (18, 22)]
        early, late = (((2, 8), (8, 14)) if p >= 0.004 and 8 in depths
                       else ((2, 10), (14, 22)))
        all_idx = np.arange(n_seq)
        curv = {}
        for name in main:
            k = names.index(name)
            for lo, hi in windows:
                point = _window_rate(fid[k], all_idx, depths, lo, hi)
                boots = [_window_rate(fid[k], idx, depths, lo, hi)
                         for idx in boot_idx]
                rate_rows.append({"p": p, "decoder": name,
                                  "window": f"{lo}-{hi}", "rate": point,
                                  "rate_se_boot": np.std(boots, ddof=1)})
            point = (_window_rate(fid[k], all_idx, depths, *late)
                     / _window_rate(fid[k], all_idx, depths, *early))
            boots = np.array([_window_rate(fid[k], idx, depths, *late)
                              / _window_rate(fid[k], idx, depths, *early)
                              for idx in boot_idx])
            curv[name] = (point, boots)
        for name in main:
            point, boots = curv[name]
            delta = boots - curv["offline"][1]
            d_point = point - curv["offline"][0]
            d_se = delta.std(ddof=1)
            rate_rows.append({
                "p": p, "decoder": name,
                "window": f"curvature r({late[0]}-{late[1]})/"
                          f"r({early[0]}-{early[1]})",
                "rate": point, "rate_se_boot": boots.std(ddof=1),
                "delta_vs_offline": d_point,
                "delta_vs_offline_se_paired": d_se,
                "abs_delta_95pct_bound": abs(d_point) + 1.96 * d_se})
    root.mkdir(parents=True, exist_ok=True)
    tag = args.terminal
    for name, rows in ((f"summary_{tag}.csv", summary_rows),
                       (f"fits_{tag}.csv", fit_rows),
                       (f"paired_vs_offline_{tag}.csv", paired_rows),
                       (f"local_rates_{tag}.csv", rate_rows),
                       (f"tie_rules_{tag}.csv", tie_rows)):
        _write_csv(root / name, rows)
    return 0


# ---------------------------------------------------------------------------
# perslot: ground-truth frame errors round by round
# ---------------------------------------------------------------------------

PERSLOT_LAGGED = [("smooth_offline", None), ("lag_w0", 0), ("lag_w1", 1),
                  ("lag_w2", 2), ("lag_w4", 4)]


def _perslot_task(args):
    p, seq_index, sequence, depth, shots, seed = args
    lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
    dem = lc.circuit.detector_error_model()
    slots = M.lrb_detector_slots(lc)
    base = M.TransferMatrixMLD(dem, slots)
    dets, obs, errs = dem.compile_sampler(seed=seed).sample(
        shots, return_errors=True)
    b_true, l_true = D.true_boundary_states(dem, slots, errs)
    names = [n for n, _ in PERSLOT_LAGGED] + [n for n, _, _ in CAUSAL]
    l_err = np.zeros((len(names), base.num_slots), dtype=np.int64)
    b_err = np.zeros((len(names), base.num_slots), dtype=np.int64)
    probe = D.CausalWindowMLD(None, None, 0, base=base)
    for k, (name, w) in enumerate(PERSLOT_LAGGED):
        marg = probe.lagged_marginals(dets, w)
        for t, post in enumerate(marg):
            pl = post.sum(axis=2)
            l_hat = (pl[:, 1] > pl[:, 0]).astype(np.int64)
            b_hat = post.sum(axis=1).argmax(axis=1)
            l_err[k, t] = int((l_hat != l_true[:, t]).sum())
            b_err[k, t] = int((b_hat != b_true[:, t]).sum())
    for j, (name, commit, w) in enumerate(CAUSAL):
        k = len(PERSLOT_LAGGED) + j
        dec = D.CausalWindowMLD(None, None, w, commit, base=base)
        _, l_hist, b_hist = dec.decode(dets, return_history=True)
        l_err[k] = (l_hist != l_true).sum(axis=0)
        b_err[k] = (b_hist != b_true).sum(axis=0)
    return p, seq_index, names, l_err, b_err


def cmd_perslot(args) -> int:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    sequences = E.clifford_sequences(30, 22, SEQ_SEED)
    probs = [float(x) for x in args.p.split(",")]
    tasks = [(p, s, sequences[s], args.depth, args.shots,
              91_000 + 1000 * s + int(p * 1e6))
             for p in probs for s in range(args.num_seq)]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_perslot_task, t)
                                 for t in tasks]):
            p, s, names, l_err, b_err = fut.result()
            results[(p, s)] = (l_err, b_err)
            print(f"perslot p={p} seq={s} done", flush=True)
    rows = []
    for p in probs:
        l_all = np.stack([results[(p, s)][0] for s in range(args.num_seq)])
        b_all = np.stack([results[(p, s)][1] for s in range(args.num_seq)])
        frac_l = l_all / args.shots
        frac_b = b_all / args.shots
        for k, name in enumerate(names):
            for t in range(l_all.shape[2]):
                label = (str(t) if t < args.depth else
                         "terminal" if t == args.depth else "final_data")
                rows.append({
                    "p": p, "estimator": name, "slot": t, "round": label,
                    "frame_error": frac_l[:, k, t].mean(),
                    "frame_error_se": frac_l[:, k, t].std(ddof=1)
                    / np.sqrt(args.num_seq),
                    "attribution_error": frac_b[:, k, t].mean(),
                    "attribution_error_se": frac_b[:, k, t].std(ddof=1)
                    / np.sqrt(args.num_seq)})
    _write_rows(root / f"perslot_depth{args.depth}.csv", rows)

    # Bulk per-round accumulation rate of the frame error, -d ln(1-2e_t)/dt
    # fitted over forward slots 2..depth-2, with a sequence bootstrap that is
    # paired across estimators (same resampled sequences).
    raw = root / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(777)
    slots = np.arange(2, args.depth - 1)
    rate_rows = []
    for p in probs:
        l_all = np.stack([results[(p, s)][0] for s in range(args.num_seq)])
        b_all = np.stack([results[(p, s)][1] for s in range(args.num_seq)])
        np.savez_compressed(raw / f"perslot_depth{args.depth}_p{p:g}.npz",
                            l_err=l_all, b_err=b_all, shots=args.shots,
                            names=np.array(names))

        def bulk(idx):
            e = l_all[idx].sum(axis=0) / (len(idx) * args.shots)
            fid = 1 - 2 * e[:, slots]
            return np.array([-np.polyfit(slots, np.log(f), 1)[0]
                             for f in fid])

        point = bulk(np.arange(args.num_seq))
        boots = np.array([bulk(rng.integers(0, args.num_seq, args.num_seq))
                          for _ in range(args.boot)])
        ref = names.index("smooth_offline")
        for k, name in enumerate(names):
            ratio = boots[:, k] / boots[:, ref]
            rate_rows.append({
                "p": p, "estimator": name, "bulk_rate": point[k],
                "bulk_rate_se_boot": boots[:, k].std(ddof=1),
                "ratio_vs_smoothing": point[k] / point[ref],
                "ratio_se_paired_boot": ratio.std(ddof=1),
                "final_error": l_all[:, k, -1].sum()
                / (args.num_seq * args.shots)})
    _write_rows(root / f"perslot_rates_depth{args.depth}.csv", rate_rows)
    return 0


# ---------------------------------------------------------------------------
# frame
# ---------------------------------------------------------------------------

def _plain_observable_circuit(seg):
    plain = seg.concatenated()
    total = plain.num_measurements
    import stim
    plain.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(seg.readout_records[q] - total)
                  for q in (0, 3)], 0)
    return plain


def cmd_frame(args) -> int:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    sequences = E.clifford_sequences(30, 22, SEQ_SEED)
    probs = [float(x) for x in args.p.split(",")]
    depths = [int(d) for d in args.depths.split(",")]

    # 1. Rule A exact: Stim classically controlled Paulis versus frame.
    rows = []
    for p in probs:
        for s in range(args.num_seq):
            for depth in depths:
                seg = D.build_segmented_lrb(sequences[s], depth, p)
                c = D.observable_flip_probability(
                    _plain_observable_circuit(seg))
                for table_name, table in (("A", D.COVARIANT_TABLE),
                                          ("A'", D.NONCOVARIANT_TABLE)):
                    phys, frame = D.rule_a_feedback_circuits(seg,
                                                             table=table)
                    a = D.observable_flip_probability(phys)
                    b = D.observable_flip_probability(frame)
                    row = {"rule": table_name, "p": p, "seq": s,
                           "depth": depth,
                           "P_flip_physical_feedback": a, "P_flip_frame": b,
                           "abs_diff": abs(a - b),
                           "P_flip_no_correction": c}
                    for label, q in (("q=0.1p", 0.1 * p), ("q=p", p)):
                        noisy, _ = D.rule_a_feedback_circuits(
                            seg, correction_noise=q, table=table)
                        row[f"P_flip_physical_noisy_{label}"] = (
                            D.observable_flip_probability(noisy))
                    rows.append(row)
        print(f"rule A exact p={p} done", flush=True)
    _write_rows(root / "frame_rule_a_exact.csv", rows)

    # 2. Identical-noise FlipSimulator: physical versus frame, shot by shot.
    rules = [("A", D.LookupEveryRound),
             ("A'", lambda: D.LookupEveryRound(D.NONCOVARIANT_TABLE)),
             ("B1", D.ConfirmedLookup),
             ("B2_w0", lambda: D.CausalMLDLogical(0)),
             ("B2_w1", lambda: D.CausalMLDLogical(1))]
    rows = []
    for p in probs:
        for s in range(min(args.num_seq, args.flip_seqs)):
            for depth in depths:
                seg = D.build_segmented_lrb(sequences[s], depth, p)
                for label, factory in rules:
                    seed = 1000 * s + 17 * depth + int(p * 1e6)
                    phys = D.simulate_feedback(seg, factory(), args.shots,
                                               seed, "physical")
                    frame = D.simulate_feedback(seg, factory(), args.shots,
                                                seed, "frame")
                    mism = int((phys.logical_flip
                                != frame.logical_flip).sum())
                    corr_shots = int(np.any(
                        [c.any(axis=1) for c in phys.corrections],
                        axis=0).sum())
                    syn_diff = int(sum((pr != fr).any(axis=1).sum()
                                       for pr, fr in zip(phys.raw_seen,
                                                         frame.raw_seen)))
                    row = {"rule": label, "p": p, "seq": s, "depth": depth,
                           "shots": args.shots, "shot_mismatches": mism,
                           "shots_with_correction": corr_shots,
                           "err_physical": phys.logical_flip.mean(),
                           "err_frame": frame.logical_flip.mean(),
                           "adjusted_syndrome_mismatch_rows": syn_diff,
                           # Measured (unadjusted) frame-run syndromes differ
                           # from the physical run's whenever a
                           # syndrome-changing correction was applied.
                           "measured_syndrome_rows_differing":
                               _unadjusted_differences(seg, frame, phys)}
                    for qlabel, q in (("0.1p", 0.1 * p), ("p", p)):
                        noisy = D.simulate_feedback(
                            seg, factory(), args.shots, seed, "physical",
                            correction_noise=q,
                            correction_noise_seed=seed + 1)
                        diff = noisy.logical_flip.astype(int) - \
                            frame.logical_flip.astype(int)
                        row[f"err_physical_noisy_q={qlabel}"] = (
                            noisy.logical_flip.mean())
                        row[f"delta_noisy_q={qlabel}"] = diff.mean()
                        row[f"delta_noisy_q={qlabel}_se"] = (
                            np.sqrt(np.abs(diff).sum()) / args.shots)
                    rows.append(row)
            print(f"flipsim p={p} seq={s} done", flush=True)
    _write_rows(root / "frame_flipsim.csv", rows)
    return 0


def _unadjusted_differences(seg, frame_run, phys_run):
    """Rounds x shots where the frame run's measured syndrome (before the
    controller XORs in the frame) differs from the physical run's."""
    tracker = D.FrameTracker(len(frame_run.logical_flip))
    count = 0
    for k in range(seg.num_rounds):
        tracker.propagate(seg.pre_round_matrices[k])
        measured = frame_run.raw_seen[k] ^ tracker.syndrome()
        count += int((measured != phys_run.raw_seen[k]).any(axis=1).sum())
        tracker.multiply(frame_run.corrections[k])
    return count


def _write_rows(path, rows):
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: (f"{v:.10g}" if isinstance(v, float) else v)
                             for k, v in row.items()})
    print(f"wrote {path} ({len(rows)} rows)", flush=True)


# ---------------------------------------------------------------------------
# toys
# ---------------------------------------------------------------------------

def cmd_toys(args) -> int:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for eps in (0.01, 0.05, 0.1):
        out = D.nonclifford_toy(eps)
        rows.append({"model": "nonclifford_T_then_Y_readout", **{
            k: v for k, v in out.items()}})
    for basis, seep in (("Z", 1), ("Z", 0), ("X", 0)):
        for eps, leak in ((0.1, 0.01), (0.1, 0.1)):
            out = D.leakage_toy(eps, leak, seep, basis)
            rows.append({"model": f"leakage_{basis}_basis_return_to_{seep}"
                         if basis == "Z" else "leakage_X_basis_coherent",
                         **out})
    _write_rows(root / "toys.csv", rows)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("grid")
    g.add_argument("--out", default=str(DEFAULT_OUT))
    g.add_argument("--p", default="0.0005,0.002,0.005",
                   help="comma list of p or p:shots")
    g.add_argument("--depths", default="all")
    g.add_argument("--num-seq", type=int, default=30)
    g.add_argument("--shots", type=int, default=40_000)
    g.add_argument("--seed", type=int, default=4242)
    g.add_argument("--workers", type=int, default=4)
    g.add_argument("--terminal", default="data5")
    g.add_argument("--resume", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--out", default=str(DEFAULT_OUT))
    r.add_argument("--boot", type=int, default=400)
    r.add_argument("--terminal", default="data5")
    f = sub.add_parser("frame")
    f.add_argument("--out", default=str(DEFAULT_OUT))
    f.add_argument("--p", default="0.0005,0.002,0.005")
    f.add_argument("--depths", default="0,2,6,10,22")
    f.add_argument("--num-seq", type=int, default=30)
    f.add_argument("--flip-seqs", type=int, default=3)
    f.add_argument("--shots", type=int, default=20_000)
    t = sub.add_parser("toys")
    t.add_argument("--out", default=str(DEFAULT_OUT))
    ps = sub.add_parser("perslot")
    ps.add_argument("--out", default=str(DEFAULT_OUT))
    ps.add_argument("--p", default="0.0005,0.002,0.005")
    ps.add_argument("--depth", type=int, default=22)
    ps.add_argument("--num-seq", type=int, default=10)
    ps.add_argument("--shots", type=int, default=20_000)
    ps.add_argument("--workers", type=int, default=4)
    ps.add_argument("--boot", type=int, default=400)
    args = parser.parse_args()
    return {"grid": cmd_grid, "report": cmd_report, "frame": cmd_frame,
            "toys": cmd_toys, "perslot": cmd_perslot}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
