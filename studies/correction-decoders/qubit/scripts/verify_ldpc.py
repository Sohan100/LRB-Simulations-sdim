"""Adversarial verification of work package C (BP+OSD, BP+LSD, union-find, sliding-window LUT).

Independent re-checks of the claims in
``studies/correction-decoders/qubit/notes/C-bposd-uf-lut.md``. To avoid
inheriting the implementer's mistakes, this script

* parses every DEM itself (one check-matrix column per Stim error
  instruction, XOR semantics for repeated targets) instead of using
  ``decoders_ldpc.dem_check_matrices``;
* builds the ldpc 2.4.1 decoders directly (``BpOsdDecoder``,
  ``BpLsdDecoder``), with explicitly listed settings;
* computes the single-fault classes ``P_D(c)`` itself;
* uses FIXED-length shot prefixes for per-syndrome decoders (no
  data-dependent stopping rule);
* uses new Clifford sequences (seed ``--seq-seed``, default 777001) and new
  sampler seeds.

The lead's ``mld.TransferMatrixMLD`` is the exact reference. The implementer's
``decoders_lut.SlidingWindowLUT`` and ``decoders_ldpc.DelfosseNickersonUF`` /
``MatchingControl`` are the objects under test, so they are imported.

Subcommands (outputs go to ``--out``, one JSON file per subcommand):

``claim2``   ldpc's ``detector_error_model_to_check_matrices`` and the
             ``SinterBpOsdDecoder`` file path on two-mechanism toy DEMs in both
             orders; census of two-class detector sets on the grid DEMs; the
             first-order cost of the last-listed labels alone.
``claim4``   every distinct single-mechanism syndrome of one circuit decoded by
             ``BpOsdDecoder`` / ``BpLsdDecoder`` in several settings (true
             ldpc defaults, serial, the implementer's settings, sinter's
             defaults) and compared with exact MLD and the first-order argmax.
``lutfo``    LUT decisions on every single-fault syndrome of NEW sequences at
             p = 5e-4 and at small p (first-order exactness by window).
``sampled``  new-seed sampled comparison against MLD: LUT w = 2..4 on all
             shots; optional per-syndrome decoders on a fixed shot prefix.
             Posterior-weighted and raw paired excess, MLD calibration.
``timing``   like-for-like timing: batch throughput per shot (with syndrome
             de-duplication for per-syndrome decoders), single-shot latency,
             and per-distinct-syndrome cost on lexicographic vs random samples.

Run (see results/ALLOCATION.md)::

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
    python studies/correction-decoders/qubit/scripts/verify_ldpc.py claim2 --out <dir>
    python studies/correction-decoders/qubit/scripts/verify_ldpc.py claim4 --out <dir> --workers 12
    python studies/correction-decoders/qubit/scripts/verify_ldpc.py lutfo --out <dir> --workers 12
    python studies/correction-decoders/qubit/scripts/verify_ldpc.py sampled --out <dir> --workers 16
    python studies/correction-decoders/qubit/scripts/verify_ldpc.py timing --out <dir>
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(v) != "1" for v in _THREAD_VARS):
    for v in _THREAD_VARS:
        os.environ[v] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np  # noqa: E402
import stim  # noqa: E402
from scipy.sparse import csc_matrix  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

GRID_SEQ_SEED = 20261028  # the implementer's evaluation sequences


# ---------------------------------------------------------------------------
# Independent DEM parsing
# ---------------------------------------------------------------------------

def parse_dem(dem: stim.DetectorErrorModel):
    """One entry per error instruction: (detectors tuple, flips L0, probability).

    Repeated targets cancel (XOR); ``^`` separators are ignored.
    """
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets: set = set()
        flips = False
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets ^= {t.val}
            elif t.is_logical_observable_id() and t.val == 0:
                flips = not flips
        out.append((tuple(sorted(dets)), flips, inst.args_copy()[0]))
    return out


def xor_p(a: float, b: float) -> float:
    return a + b - 2.0 * a * b


def check_matrix(mechs, num_dets):
    """(H csc uint8, L row uint8, priors) with one column per mechanism.

    Mechanisms without detectors are dropped (none exist for data5)."""
    rows, cols, lrow, priors = [], [], [], []
    j = 0
    for dets, flips, prob in mechs:
        if not dets:
            continue
        for d in dets:
            rows.append(d)
            cols.append(j)
        lrow.append(int(flips))
        priors.append(prob)
        j += 1
    h = csc_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                   shape=(num_dets, j), dtype=np.uint8)
    return h, np.array(lrow, dtype=np.int64), np.array(priors)


def single_fault_sets(mechs):
    """{detector set: [P(class 0), P(class 1), n_mechanisms, last_listed_class]}."""
    sets: dict = {}
    for dets, flips, prob in mechs:
        if not dets:
            continue
        entry = sets.setdefault(dets, [0.0, 0.0, 0, None])
        entry[int(flips)] = xor_p(entry[int(flips)], prob)
        entry[2] += 1
        entry[3] = int(flips)
    return sets


def events_for_sets(keys, num_dets):
    events = np.zeros((len(keys), num_dets), dtype=bool)
    for i, dets in enumerate(keys):
        events[i, list(dets)] = True
    return events


def build(seq, depth, p):
    lc = Q.build_lrb_circuit(seq, depth, p, terminal="data5")
    dem = lc.circuit.detector_error_model()
    return lc, dem


# ---------------------------------------------------------------------------
# claim 2
# ---------------------------------------------------------------------------

def _sinter_predict(dem: stim.DetectorErrorModel, syndrome: np.ndarray):
    """Decode one syndrome through SinterBpOsdDecoder.decode_via_files (defaults)."""
    from ldpc.sinter_decoders import SinterBpOsdDecoder
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        dem.to_file(tmp / "dem.dem")
        stim.write_shot_data_file(data=syndrome[None, :].astype(bool),
                                  path=str(tmp / "dets.b8"), format="b8",
                                  num_detectors=dem.num_detectors)
        dec = SinterBpOsdDecoder()
        dec.decode_via_files(num_shots=1, num_dets=dem.num_detectors,
                             num_obs=dem.num_observables,
                             dem_path=tmp / "dem.dem",
                             dets_b8_in_path=tmp / "dets.b8",
                             obs_predictions_b8_out_path=tmp / "obs.b8",
                             tmp_dir=tmp)
        pred = stim.read_shot_data_file(path=str(tmp / "obs.b8"), format="b8",
                                        num_observables=dem.num_observables)
        settings = dict(bp_method=dec.bposd.bp_method,
                        max_iter=dec.bposd.max_iter,
                        schedule=dec.bposd.schedule,
                        osd_method=dec.bposd.osd_method,
                        osd_order=dec.bposd.osd_order,
                        ms_scaling_factor=dec.bposd.ms_scaling_factor)
    return bool(pred[0, 0]), settings


def claim2(args) -> dict:
    from ldpc.ckt_noise.dem_matrices import (
        detector_error_model_to_check_matrices)
    out: dict = {"toy": [], "census": [], "label_cost": []}
    # (a) Two mechanisms with the same detector set, opposite observable.
    toys = {
        "less_probable_last": "error(0.3) D0 D1\nerror(0.1) D0 D1 L0\n",
        "more_probable_last": "error(0.1) D0 D1 L0\nerror(0.3) D0 D1\n",
        "flip_class_more_probable_last": "error(0.1) D0 D1\nerror(0.3) D0 D1 L0\n",
        "three_mechanisms": ("error(0.3) D0 D1\nerror(0.1) D0 D1 L0\n"
                             "error(0.05) D1\n"),
    }
    for name, text in toys.items():
        dem = stim.DetectorErrorModel(text)
        m = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True)
        h = m.check_matrix.toarray()
        lmat = m.observables_matrix.toarray()
        cols = []
        for j in range(h.shape[1]):
            cols.append({"detectors": np.flatnonzero(h[:, j]).tolist(),
                         "L0": int(lmat[0, j]),
                         "prior": float(m.priors[j])})
        syndrome = np.array([1, 1] + [0] * (dem.num_detectors - 2), np.uint8)
        # Exact posterior by brute force over the mechanisms.
        mechs = parse_dem(dem)
        joint = {0: 0.0, 1: 0.0}
        for picks in np.ndindex(*(2,) * len(mechs)):
            w, dets, flip = 1.0, set(), 0
            for pick, (d, f, pr) in zip(picks, mechs):
                w *= pr if pick else 1.0 - pr
                if pick:
                    dets ^= set(d)
                    flip ^= int(f)
            if dets == set(np.flatnonzero(syndrome).tolist()):
                joint[flip] += w
        sinter_pred, settings = _sinter_predict(dem, syndrome)
        out["toy"].append({
            "dem": name, "text": text.strip(), "ldpc_columns": cols,
            "exact_P(L=1|s)": joint[1] / (joint[0] + joint[1]),
            "mld_flip": bool(joint[1] > joint[0]),
            "sinter_bposd_flip": sinter_pred,
            "sinter_bposd_settings": settings})
    # (b) Census on the implementer's grid DEMs (sequences 0-9).
    seqs = E.clifford_sequences(10, 22, GRID_SEQ_SEED)
    for depth in (0, 2, 6, 14, 22):
        for p in (5e-4, 1e-2):
            for si, seq in enumerate(seqs):
                lc, dem = build(seq, depth, p)
                mechs = parse_dem(dem)
                sets = single_fault_sets(mechs)
                multi = [v for v in sets.values() if v[2] > 1]
                two_class = [v for v in sets.values()
                             if v[0] > 0 and v[1] > 0]
                pair_opposite = all(v[2] == 2 and v[0] > 0 and v[1] > 0
                                    for v in multi)
                # ldpc's merged column observable vs last-listed class.
                m = detector_error_model_to_check_matrices(
                    dem, allow_undecomposed_hyperedges=True)
                h = m.check_matrix.tocsc()
                lmat = m.observables_matrix.toarray()
                ldpc_class = {}
                for j in range(h.shape[1]):
                    key = tuple(sorted(h.indices[h.indptr[j]:h.indptr[j + 1]]
                                       .tolist()))
                    ldpc_class[key] = int(lmat[0, j])
                last_ok = all(ldpc_class[k] == v[3] for k, v in sets.items())
                wrong_label = [k for k, v in sets.items()
                               if v[0] > 0 and v[1] > 0
                               and v[ldpc_class[k]] < v[1 - ldpc_class[k]]]
                cost = sum(abs(sets[k][0] - sets[k][1]) for k in wrong_label)
                floor = sum(min(v[0], v[1]) for v in sets.values())
                out["census"].append({
                    "p": p, "depth": depth, "seq": si, "num_sets": len(sets),
                    "num_mechanisms": len(mechs),
                    "two_class_sets": len(two_class),
                    "frac_two_class": len(two_class) / len(sets),
                    "all_multi_sets_are_opposite_pairs": pair_opposite,
                    "ldpc_class_equals_last_listed": last_ok,
                    "two_class_sets_with_less_probable_label":
                        len(wrong_label),
                    "label_cost_over_p": cost / p,
                    "mld_floor_over_p": floor / p})
    return out


# ---------------------------------------------------------------------------
# claim 4
# ---------------------------------------------------------------------------

BP_CONFIGS = {
    # ldpc 2.4.1 BpOsdDecoder(H, error_channel=...) with nothing else set:
    # product_sum (the docstring says minimum_sum), max_iter = n, parallel, OSD_0.
    "osd_ldpc_defaults": ("osd", {}),
    "osd_ldpc_defaults_serial": ("osd", {"schedule": "serial"}),
    # The implementer's bposd_cs7 / bposd_cs7_serial / bposd_0.
    "osd_cs7_ps_it5_parallel": ("osd", dict(bp_method="product_sum", max_iter=5,
                                            schedule="parallel",
                                            osd_method="OSD_CS", osd_order=7)),
    "osd_cs7_ps_it5_serial": ("osd", dict(bp_method="product_sum", max_iter=5,
                                          schedule="serial",
                                          osd_method="OSD_CS", osd_order=7)),
    "osd_cs7_ps_itn_parallel": ("osd", dict(bp_method="product_sum",
                                            max_iter=0, schedule="parallel",
                                            osd_method="OSD_CS", osd_order=7)),
    # Same OSD-CS7 post-processing with min-sum BP (is the failure product-sum specific?).
    "osd_cs7_ms0625_it5_parallel": ("osd", dict(bp_method="minimum_sum",
                                                ms_scaling_factor=0.625,
                                                max_iter=5,
                                                schedule="parallel",
                                                osd_method="OSD_CS",
                                                osd_order=7)),
    "osd_cs7_ms1_it5_parallel": ("osd", dict(bp_method="minimum_sum",
                                             ms_scaling_factor=1.0, max_iter=5,
                                             schedule="parallel",
                                             osd_method="OSD_CS",
                                             osd_order=7)),
    "osd_cs7_ms0625_itn_parallel": ("osd", dict(bp_method="minimum_sum",
                                                ms_scaling_factor=0.625,
                                                max_iter=0,
                                                schedule="parallel",
                                                osd_method="OSD_CS",
                                                osd_order=7)),
    # SinterBpOsdDecoder's own defaults (min-sum 0.625, max_iter = n, OSD_0).
    "osd_sinter_defaults": ("osd", dict(bp_method="ms", ms_scaling_factor=0.625,
                                        max_iter=0, schedule="parallel",
                                        osd_method="osd0", osd_order=0)),
    "osd_sinter_defaults_serial": ("osd", dict(bp_method="ms",
                                               ms_scaling_factor=0.625,
                                               max_iter=0, schedule="serial",
                                               osd_method="osd0",
                                               osd_order=0)),
    # The implementer's bplsd_cs7 and bplsd_cs7_always.
    "lsd_cs7_ps_it5_parallel": ("lsd", dict(bp_method="product_sum",
                                            max_iter=5, schedule="parallel",
                                            lsd_method="LSD_CS", lsd_order=7,
                                            bits_per_step=1)),
    "lsd_cs7_ps_it5_always": ("lsd", dict(bp_method="product_sum", max_iter=5,
                                          schedule="parallel",
                                          lsd_method="LSD_CS", lsd_order=7,
                                          bits_per_step=1,
                                          always_run_lsd=True)),
}


def make_bp_decoder(h, priors, config: str):
    kind, kwargs = BP_CONFIGS[config]
    if kind == "osd":
        from ldpc import BpOsdDecoder
        return BpOsdDecoder(h, error_channel=[float(x) for x in priors],
                            **kwargs)
    from ldpc import BpLsdDecoder
    return BpLsdDecoder(h, error_channel=[float(x) for x in priors], **kwargs)


def claim4_task(task: dict) -> dict:
    seq, depth, p = task["sequence"], task["depth"], task["p"]
    lc, dem = build(seq, depth, p)
    mechs = parse_dem(dem)
    sets = single_fault_sets(mechs)
    if task.get("columns", "mechanism") == "dominant":
        # One column per detector set: XOR-combined prior, more probable class.
        dom = [(k, v[1] > v[0], xor_p(v[0], v[1])) for k, v in sets.items()]
        h, lrow, priors = check_matrix(dom, dem.num_detectors)
    else:
        h, lrow, priors = check_matrix(mechs, dem.num_detectors)
    keys = list(sets)
    probs = np.array([[sets[k][0], sets[k][1]] for k in keys])
    two_class = (probs > 0).all(axis=1)
    events = events_for_sets(keys, dem.num_detectors)
    mld = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    mld_flip = mld.log_likelihood_ratio(events) < 0
    argmax = probs[:, 1] > probs[:, 0]
    floor = probs.min(axis=1).sum()
    res = {"seq_label": task["seq_label"], "depth": depth, "p": p,
           "num_columns": int(h.shape[1]), "num_sets": len(keys),
           "num_two_class": int(two_class.sum()),
           "mld_off_argmax": int((mld_flip != argmax).sum()),
           "mld_off_argmax_one_class": int(
               ((mld_flip != argmax) & ~two_class).sum()),
           "mld_floor_over_p": floor / p, "configs": {}}
    for config in task["configs"]:
        dec = make_bp_decoder(h, priors, config)
        pred = np.zeros(len(keys), dtype=bool)
        conv = np.zeros(len(keys), dtype=bool)
        sat = np.ones(len(keys), dtype=bool)
        t0 = time.perf_counter()
        for i in range(len(keys)):
            s = events[i].astype(np.uint8)
            e = np.asarray(dec.decode(s), dtype=np.int64)
            conv[i] = bool(dec.converge)
            pred[i] = (lrow @ e) % 2 == 1
            sat[i] = np.array_equal((h @ e) % 2, s.astype(np.int64))
        seconds = time.perf_counter() - t0
        wrong_mass = np.where(pred, probs[:, 0], probs[:, 1]).sum()
        off_mld = pred != mld_flip
        off_arg = pred != argmax
        info = {}
        for label, mask in (("one_class", ~two_class), ("two_class",
                                                          two_class)):
            info[label] = {
                "sets": int(mask.sum()),
                "off_mld": int((off_mld & mask).sum()),
                "off_argmax": int((off_arg & mask).sum()),
                "off_argmax_and_bp_converged": int(
                    (off_arg & mask & conv).sum()),
                "excess_over_p": float(
                    (np.where(pred, probs[:, 0], probs[:, 1])
                     - probs.min(axis=1))[mask].sum() / p)}
        res["configs"][config] = {
            "settings": {"bp_method": dec.bp_method,
                         "max_iter": int(dec.max_iter),
                         "schedule": dec.schedule},
            "excess_over_p": float((wrong_mass - floor) / p),
            "bp_converged_frac": float(conv.mean()),
            "unsatisfied": int((~sat).sum()),
            "seconds": seconds, **info}
    return res


def claim4(args) -> dict:
    grid = E.clifford_sequences(10, 22, GRID_SEQ_SEED)
    fresh = E.clifford_sequences(args.num_new, 22, args.seq_seed)
    configs = list(BP_CONFIGS) if not args.configs else args.configs.split(",")
    tasks = []
    for depth in [int(x) for x in args.depths.split(",")]:
        for si in range(args.num_grid):
            tasks.append({"sequence": grid[si], "seq_label": f"grid{si}",
                          "depth": depth, "p": args.p, "configs": configs,
                          "columns": args.columns})
        for si in range(args.num_new):
            tasks.append({"sequence": fresh[si], "seq_label": f"new{si}",
                          "depth": depth, "p": args.p, "configs": configs,
                          "columns": args.columns})
    # Each finished task is also appended to a JSON-lines file, so a run cut
    # short by the allocation still leaves its completed circuits.
    partial = Path(args.out) / f"claim4{args.tag}.partial.jsonl"
    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool, \
            open(partial, "a") as sink:
        for res in pool.map(claim4_task, tasks):
            results.append(res)
            sink.write(json.dumps(res, default=float) + "\n")
            sink.flush()
    return {"p": args.p, "results": results}


# ---------------------------------------------------------------------------
# LUT first order
# ---------------------------------------------------------------------------

def lutfo_task(task: dict) -> dict:
    from lrb_qubit import decoders_lut as DT
    seq, depth, p = task["sequence"], task["depth"], task["p"]
    lc, dem = build(seq, depth, p)
    sets = single_fault_sets(parse_dem(dem))
    keys = list(sets)
    probs = np.array([[sets[k][0], sets[k][1]] for k in keys])
    events = events_for_sets(keys, dem.num_detectors)
    argmax = probs[:, 1] > probs[:, 0]
    floor = probs.min(axis=1).sum()
    gap = np.abs(probs[:, 1] - probs[:, 0]) / probs.sum(axis=1)
    out = {"seq_label": task["seq_label"], "depth": depth, "p": p,
           "num_sets": len(keys), "mld_floor_over_p": floor / p,
           "decoders": {}}
    slots = M.lrb_detector_slots(lc)
    preds = {"mld": M.TransferMatrixMLD(dem, slots).log_likelihood_ratio(
        events) < 0}
    for w in task["windows"]:
        lut = DT.SlidingWindowLUT(dem, slots, w)
        preds[f"lut_w{w}"] = lut.predict(events)
    for name, pred in preds.items():
        wrong = np.where(pred, probs[:, 0], probs[:, 1]).sum()
        off = pred != argmax
        out["decoders"][name] = {
            "excess_over_p": float((wrong - floor) / p),
            "sets_off_argmax": int(off.sum()),
            "off_argmax_min_rel_gap": float(gap[off].min()) if off.any()
            else None,
            "off_argmax_max_rel_gap": float(gap[off].max()) if off.any()
            else None}
    return out


def lutfo(args) -> dict:
    fresh = E.clifford_sequences(args.num_new, 22, args.seq_seed)
    tasks = []
    for p in [float(x) for x in args.p_list.split(",")]:
        for depth in [int(x) for x in args.depths.split(",")]:
            for si in range(args.num_new):
                tasks.append({"sequence": fresh[si], "seq_label": f"new{si}",
                              "depth": depth, "p": p,
                              "windows": [int(w) for w in
                                          args.windows.split(",")]})
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lutfo_task, tasks))
    return {"results": results}


# ---------------------------------------------------------------------------
# Sampled comparison
# ---------------------------------------------------------------------------

def paired_stats(pred, flip, mld_flip, llr) -> dict:
    pred = np.asarray(pred, dtype=bool)
    gap = np.abs(llr)
    disagree = pred != mld_flip
    wrong = pred != flip
    mld_wrong = mld_flip != flip
    diff = np.where(disagree, np.tanh(gap / 2.0), 0.0)
    return {"n": int(pred.size), "errors": int(wrong.sum()),
            "n10": int((wrong & ~mld_wrong).sum()),
            "n01": int((~wrong & mld_wrong).sum()),
            "disagree": int(disagree.sum()),
            "rb_diff": float(diff.sum()), "rb_diff2": float((diff ** 2).sum())}


def decode_per_syndrome(decode_fn, events: np.ndarray, lrow) -> np.ndarray:
    """Decode each distinct non-zero syndrome once; broadcast the L0 flip."""
    packed = np.packbits(events, axis=1)
    view = np.ascontiguousarray(packed).view(
        np.dtype((np.void, packed.shape[1])))[:, 0]
    _, first, inverse = np.unique(view, return_index=True, return_inverse=True)
    rows = events[first]
    out = np.zeros(rows.shape[0], dtype=bool)
    for i, row in enumerate(rows):
        if row.any():
            e = np.asarray(decode_fn(row.astype(np.uint8)), dtype=np.int64)
            out[i] = (lrow @ e) % 2 == 1
    return out[inverse.reshape(-1)], int(rows.shape[0])


def sampled_task(task: dict) -> dict:
    from lrb_qubit import decoders_lut as DT
    seq, depth, p = task["sequence"], task["depth"], task["p"]
    lc, dem = build(seq, depth, p)
    slots = M.lrb_detector_slots(lc)
    mld = M.TransferMatrixMLD(dem, slots)
    sampler = lc.circuit.compile_detector_sampler(seed=task["seed"])
    dets, obs = sampler.sample(task["shots"], separate_observables=True)
    flip = obs[:, 0]
    llr = mld.log_likelihood_ratio(dets)
    mld_flip = llr < 0
    gap = np.abs(llr)
    with np.errstate(over="ignore"):
        p_small = 1.0 / (1.0 + np.exp(gap))
    res = {"seq_label": task["seq_label"], "depth": depth, "p": p,
           "shots": int(flip.size), "mld_errors": int((mld_flip != flip).sum()),
           "mld_post_sum": float(p_small.sum()),
           "mld_post_var": float((p_small * (1 - p_small)).sum()),
           "decoders": {}}
    for w in task["windows"]:
        lut = DT.SlidingWindowLUT(dem, slots, w)
        res["decoders"][f"lut_w{w}"] = paired_stats(lut.predict(dets), flip,
                                                    mld_flip, llr)
    n = min(task["prefix"], flip.size)
    sub = dets[:n]
    mechs = parse_dem(dem)
    h, lrow, priors = check_matrix(mechs, dem.num_detectors)
    for name in task["decoders"]:
        t0 = time.perf_counter()
        if name in BP_CONFIGS:
            dec = make_bp_decoder(h, priors, name)
            pred, uniq = decode_per_syndrome(dec.decode, sub, lrow)
        elif name in ("uf_dn", "mwpm_graph"):
            from lrb_qubit import decoders_ldpc as DL
            graph = DL.graph_check_matrices(
                lc.circuit.detector_error_model(decompose_errors=True))
            if name == "uf_dn":
                uf = DL.DelfosseNickersonUF(graph)
                glrow = graph.observables_matrix.toarray()[0].astype(np.int64)
                pred, uniq = decode_per_syndrome(uf.decode, sub, glrow)
            else:
                pred = DL.MatchingControl(graph).predict(sub)
                uniq = -1
        else:
            raise ValueError(name)
        stats = paired_stats(pred, flip[:n], mld_flip[:n], llr[:n])
        stats.update({"unique": uniq, "seconds": time.perf_counter() - t0})
        res["decoders"][name] = stats
    return res


def aggregate_sampled(results) -> list[dict]:
    groups: dict = {}
    for r in results:
        for name, s in r["decoders"].items():
            groups.setdefault((r["p"], r["depth"], name), []).append(s)
        groups.setdefault((r["p"], r["depth"], "_mld_calibration"), []).append(
            {"n": r["shots"], "mld_errors": r["mld_errors"],
             "post": r["mld_post_sum"], "var": r["mld_post_var"]})
    rows = []
    for (p, depth, name), items in sorted(groups.items()):
        k = len(items)
        if name == "_mld_calibration":
            obs = sum(i["mld_errors"] for i in items)
            post = sum(i["post"] for i in items)
            var = sum(i["var"] for i in items)
            n = sum(i["n"] for i in items)
            rows.append({"p": p, "depth": depth, "decoder": "mld_calibration",
                         "num_seq": k, "shots": n,
                         "observed_ler": obs / n, "posterior_ler": post / n,
                         "z_conditional": (obs - post) / math.sqrt(var)
                         if var > 0 else 0.0})
            continue
        n = np.array([i["n"] for i in items], float)
        rbd = np.array([i["rb_diff"] for i in items]) / n
        var_rbd = np.maximum(np.array([i["rb_diff2"] for i in items]) / n
                             - rbd ** 2, 0.0) / n
        n10 = np.array([i["n10"] for i in items], float)
        n01 = np.array([i["n01"] for i in items], float)
        raw = (n10 - n01) / n
        var_raw = (n10 + n01 - (n10 - n01) ** 2 / n) / n ** 2
        dis = np.array([i["disagree"] for i in items]) / n
        rows.append({"p": p, "depth": depth, "decoder": name, "num_seq": k,
                     "shots": int(n.sum()),
                     "excess_rb_pp": 100 * rbd.mean(),
                     "excess_rb_se_pp": 100 * math.sqrt(var_rbd.sum()) / k,
                     "excess_raw_pp": 100 * raw.mean(),
                     "excess_raw_se_pp": 100 * math.sqrt(var_raw.sum()) / k,
                     "agreement_pct": 100 * (1 - dis.mean())})
    return rows


def sampled(args) -> dict:
    fresh = E.clifford_sequences(args.num_new, 22, args.seq_seed)
    points = [tuple(x.split(":")) for x in args.points.split(",")]
    decoders = [d for d in args.decoders.split(",") if d]
    tasks = []
    for pi, (p, depth) in enumerate(points):
        for si in range(args.num_new):
            tasks.append({"sequence": fresh[si], "seq_label": f"new{si}",
                          "p": float(p), "depth": int(depth),
                          "seed": args.sample_seed + 1000 * pi + si,
                          "shots": args.shots, "prefix": args.prefix,
                          "windows": [int(w) for w in args.windows.split(",")
                                      if w],
                          "decoders": decoders})
    tasks.sort(key=lambda t: -(t["depth"] + 1) * t["p"])
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(sampled_task, tasks))
    return {"results": results, "summary": aggregate_sampled(results)}


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def timing(args) -> dict:
    from lrb_qubit import decoders_lut as DT
    seq = E.clifford_sequences(1, 22, GRID_SEQ_SEED)[0]
    rng = np.random.default_rng(5)
    rows = []

    def best(fn, repeats=3):
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        return min(times)

    for p in [float(x) for x in args.p_list.split(",")]:
        for depth in [int(x) for x in args.depths.split(",")]:
            lc, dem = build(seq, depth, p)
            slots = M.lrb_detector_slots(lc)
            dets, _ = lc.circuit.compile_detector_sampler(seed=7).sample(
                50_000, separate_observables=True)
            mld = M.TransferMatrixMLD(dem, slots)
            lut = DT.SlidingWindowLUT(dem, slots, 3)
            row = {"p": p, "depth": depth}
            row["mld_batch_us_per_shot"] = 1e6 * best(
                lambda: mld.log_likelihood_ratio(dets)) / dets.shape[0]
            row["lut3_batch_us_per_shot"] = 1e6 * best(
                lambda: lut.predict(dets)) / dets.shape[0]
            single = dets[:200]
            row["mld_single_shot_us"] = 1e6 * best(
                lambda: [mld.log_likelihood_ratio(single[i:i + 1])
                         for i in range(200)]) / 200
            row["lut3_single_shot_us"] = 1e6 * best(
                lambda: [lut.predict(single[i:i + 1]) for i in range(200)]) / 200
            mechs = parse_dem(dem)
            h, lrow, priors = check_matrix(mechs, dem.num_detectors)
            packed = np.packbits(dets, axis=1)
            view = np.ascontiguousarray(packed).view(
                np.dtype((np.void, packed.shape[1])))[:, 0]
            _, first = np.unique(view, return_index=True)
            uniq = dets[first]
            uniq = uniq[uniq.any(axis=1)]
            lex = uniq[:300]  # the implementer's sample (lexicographic first)
            rnd = uniq[rng.choice(uniq.shape[0], min(300, uniq.shape[0]),
                                  replace=False)]
            row["distinct_nonzero_per_50k"] = int(uniq.shape[0])
            row["mean_weight_lex300"] = float(lex.sum(axis=1).mean())
            row["mean_weight_random300"] = float(rnd.sum(axis=1).mean())
            row["mean_weight_shots"] = float(dets.sum(axis=1).mean())
            for config in args.configs.split(","):
                dec = make_bp_decoder(h, priors, config)
                for label, sample in (("lex300", lex), ("random300", rnd)):
                    sec = best(lambda: [dec.decode(r.astype(np.uint8))
                                        for r in sample], repeats=1)
                    row[f"{config}_us_per_syndrome_{label}"] = \
                        1e6 * sec / max(sample.shape[0], 1)
                shots = dets[:args.throughput_shots]
                t0 = time.perf_counter()
                _, n_unique = decode_per_syndrome(dec.decode, shots, lrow)
                sec = time.perf_counter() - t0
                row[f"{config}_dedup_us_per_shot"] = 1e6 * sec / shots.shape[0]
                row[f"{config}_dedup_unique_frac"] = n_unique / shots.shape[0]
            rows.append(row)
            print(json.dumps(row), flush=True)
    return {"rows": rows}


# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("command", choices=["claim2", "claim4", "lutfo",
                                        "sampled", "timing"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seq-seed", type=int, default=777001)
    ap.add_argument("--num-new", type=int, default=3)
    ap.add_argument("--num-grid", type=int, default=0)
    ap.add_argument("--p", type=float, default=5e-4)
    ap.add_argument("--p-list", default="5e-4,1e-6")
    ap.add_argument("--depths", default="6")
    ap.add_argument("--windows", default="1,2,3,4")
    ap.add_argument("--configs", default="")
    ap.add_argument("--columns", default="mechanism",
                    choices=["mechanism", "dominant"],
                    help="claim4: one column per DEM mechanism, or one per "
                         "detector set with the more probable class")
    ap.add_argument("--points", default="5e-3:6")
    ap.add_argument("--decoders", default="")
    ap.add_argument("--shots", type=int, default=50_000)
    ap.add_argument("--prefix", type=int, default=5_000)
    ap.add_argument("--sample-seed", type=int, default=909090)
    ap.add_argument("--throughput-shots", type=int, default=2_000)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    result = {"claim2": claim2, "claim4": claim4, "lutfo": lutfo,
              "sampled": sampled, "timing": timing}[args.command](args)
    result["args"] = vars(args)
    result["seconds"] = time.perf_counter() - t0
    path = out / f"{args.command}{args.tag}.json"
    path.write_text(json.dumps(result, indent=1, default=float))
    print(f"wrote {path} ({result['seconds']:.1f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
