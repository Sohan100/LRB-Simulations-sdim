"""Adversarial verification of work package B (matching decoders vs exact MLD).

Independent of the implementer's module: ``lrb_qubit.decoders_matching`` and
``studies/correction-decoders/qubit/scripts/matching_vs_mld.py`` are never imported. Everything is built
from Stim 1.15, PyMatching 2.3.1 and the lead's model modules (``circuits``,
``code``, ``experiment``, ``mld``). B's result CSVs are read only to compare.

Subcommands (outputs go to ``--out``, default ``results/B-verify``):

``census``  H1 (and the first-order part of H3). Single-fault census. For every
    undecomposed DEM mechanism i (probability q_i, detectors s_i, flip l_i),
    decode s_i alone with
      - ``mwpm``: PyMatching from ``detector_error_model(decompose_errors=True)``;
      - ``mwpm_corr``: PyMatching correlated matching, built from the same DEM
        with its detector-free components removed (it refuses the raw DEM);
      - ``mld``: the lead's exact ``TransferMatrixMLD``;
      - ``none``: always predict "no flip";
      - ``classaware``: a PyMatching graph in which parallel components with
        different observables stay separate (the second class is routed
        through a dummy node that never fires), i.e. the decomposed graph
        without PyMatching's "first observable wins" merge rule (its exact
        ties are broken by PyMatching, so it can differ from B's min-sum
        version, which breaks ties to "no flip");
      - ``mwe`` / ``mwe_tieflip``: exact minimum-weight error on the
        undecomposed hypergraph (own ``MinPlusSweep``), ties broken both ways;
    and sum q_i / p over the mechanisms each decoder gets wrong. ``E1`` is the
    exact first-order MLD (= minimum-weight-error) coefficient
    sum_s min(P0(s), P1(s)) taken directly from the DEM. Per-sequence linear
    fits over depth give the error per depth step.

``h4``  Detector-free ("L0-only") components in Stim's decomposition: counts
    and mass per circuit, whether plain PyMatching warns, whether its graph is
    identical to the graph of the DEM with those components deleted (i.e. it
    drops them), and whether correlated PyMatching raises ``ValueError``.

``h6``  Physical RB error per depth step: analytic count over the 24-element
    catalog, and the exact flip probability from Stim's DEM of
    ``build_physical_rb_circuit``.

``sample``  H2 spot checks with NEW sampler seeds (PyMatching and lead MLD on
    identical shots). For m <= 2 (at most 18 detectors) also exact brute-force
    min-sum class weights on (a) the undecomposed hypergraph (MWE), (b) the
    class-aware graph and (c) PyMatching's own merged graph (its decision and
    complementary gap). That re-derives the H3 ladder and the H5 calibration
    and acceptance numbers at m <= 2 without B's min-sum transfer matrix.

Usage (shared allocation, see results/ALLOCATION.md):

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
    python studies/correction-decoders/qubit/scripts/verify_matching.py all --workers 16
"""

from __future__ import annotations

import os
import sys

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_v) != "1" for _v in _THREAD_VARS):
    for _v in _THREAD_VARS:
        os.environ[_v] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import argparse  # noqa: E402
import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from collections import Counter  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pymatching  # noqa: E402
import stim  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

B_SEQ_SEED = 20261028        # B used sequences 0..9 of this base seed
NEW_SEQ_SEED = 5_550_000      # fresh sequences for this verification
SAMPLE_SEED = 31_337_000_000  # fresh sampler seeds (B used 202610282 + ...)
B_DEPTHS = (0, 2, 6, 14, 22)
ALL_DEPTHS = tuple(range(23))
B_RESULTS = ROOT / "studies/correction-decoders/qubit/results/B"
DEFAULT_OUT = ROOT / "studies/correction-decoders/qubit/results/B-verify"
TIE = 1e-9
GAP_EDGES = np.concatenate([[0.0, TIE], np.arange(0.01, 60.0 + 1e-9, 0.01),
                            [np.inf]])
CALIB_EDGES = np.array([0.0, TIE, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0,
                        np.inf])


# ---------------------------------------------------------------------------
# DEM helpers (own implementation)
# ---------------------------------------------------------------------------

def components(instruction: stim.DemInstruction) -> list[tuple[tuple, bool]]:
    """``(sorted detectors, flips L0)`` of each ``^``-separated component."""
    comps: list[list] = [[set(), False]]
    for t in instruction.targets_copy():
        if t.is_separator():
            comps.append([set(), False])
        elif t.is_relative_detector_id():
            comps[-1][0] ^= {t.val}
        elif t.is_logical_observable_id() and t.val == 0:
            comps[-1][1] = not comps[-1][1]
    return [(tuple(sorted(d)), bool(f)) for d, f in comps]


def error_lines(dem: stim.DetectorErrorModel):
    for ins in dem.flattened():
        if ins.type == "error":
            yield ins.args_copy()[0], components(ins)


def undecomposed_mechanisms(dem) -> list[tuple[tuple, bool, float]]:
    out = []
    for q, comps in error_lines(dem):
        if len(comps) != 1:
            raise ValueError("separator in an undecomposed DEM")
        (dets, flip), = comps
        out.append((dets, flip, q))
    return out


def weight(q: float) -> float:
    return math.log((1.0 - q) / q)


def strip_detector_free(dem: stim.DetectorErrorModel):
    """Delete the components that have no detector from decomposed lines."""
    out = stim.DetectorErrorModel()
    changed = 0
    for ins in dem.flattened():
        if ins.type != "error":
            out.append(ins)
            continue
        groups: list[list] = [[]]
        for t in ins.targets_copy():
            if t.is_separator():
                groups.append([])
            else:
                groups[-1].append(t)
        keep = [g for g in groups if any(t.is_relative_detector_id() for t in g)]
        if len(keep) == len(groups):
            out.append(ins)
            continue
        changed += 1
        targets: list = []
        for k, g in enumerate(keep):
            if k:
                targets.append(stim.target_separator())
            targets.extend(g)
        if targets:
            out.append("error", ins.args_copy(), targets)
    return out, changed


def classaware_edges(dec: stim.DetectorErrorModel):
    """Graphlike components XOR-merged per (detectors, L0) class, in DEM order."""
    merged: dict = {}
    order: list = []
    for q, comps in error_lines(dec):
        for dets, flip in comps:
            if not dets:
                continue
            if len(dets) > 2:
                raise ValueError("non-graphlike component")
            key = (dets, flip)
            if key not in merged:
                merged[key] = 0.0
                order.append(key)
            r = merged[key]
            merged[key] = r * (1 - q) + q * (1 - r)
    return [(d, f, merged[(d, f)]) for d, f in order]


def classaware_matching(dec, num_detectors: int):
    """PyMatching graph with both classes of every parallel edge kept.

    The second class of a (detectors) pair is routed through a fresh dummy
    node with half the weight on each side. The dummy node never fires, so a
    matching uses both halves or neither: an exact parallel edge.
    """
    m = pymatching.Matching()
    seen: set = set()
    dummy = num_detectors
    for dets, flip, q in classaware_edges(dec):
        w = weight(q)
        fid = {0} if flip else set()
        if dets not in seen:
            seen.add(dets)
            if len(dets) == 1:
                m.add_boundary_edge(dets[0], fault_ids=fid, weight=w)
            else:
                m.add_edge(dets[0], dets[1], fault_ids=fid, weight=w)
        else:
            x = dummy
            dummy += 1
            m.add_edge(dets[0], x, fault_ids=fid, weight=w / 2)
            if len(dets) == 1:
                m.add_boundary_edge(x, fault_ids=set(), weight=w / 2)
            else:
                m.add_edge(x, dets[1], fault_ids=set(), weight=w / 2)
    return m


def pymatching_graph_edges(matching) -> list[tuple[tuple, bool, float]]:
    out = []
    for u, v, data in matching.edges():
        dets = (int(u),) if v is None else tuple(sorted((int(u), int(v))))
        out.append((dets, 0 in data["fault_ids"], float(data["weight"])))
    return out


def minsum_table(num_detectors: int, mechanisms) -> np.ndarray:
    """Brute force W(syndrome, l) = min total weight, each mechanism used <= once.

    Index: detector k in bit k, logical class in bit ``num_detectors``.
    """
    size = 2 ** (num_detectors + 1)
    table = np.full(size, np.inf)
    table[0] = 0.0
    index = np.arange(size)
    for dets, flip, w in mechanisms:
        v = 0
        for d in dets:
            v ^= 1 << int(d)
        if flip:
            v ^= 1 << num_detectors
        if v == 0:
            continue
        table = np.minimum(table, table[index ^ v] + w)
    return table


class MinPlusSweep:
    """Exact min-sum class weights by a time-slot sweep (own implementation).

    ``V_t(b, l)`` is the minimum weight of a subset of mechanisms whose latest
    slot is <= t that explains every observed detector in slots < t, leaves
    residual ``b`` on slot t and has logical parity ``l``. Mechanisms whose
    latest slot is t + 1 touch slots t and t + 1 only, so

        V_{t+1}(d, l) = min_{c, l'} V_t(s_t ^ c, l ^ l') + K_{t+1}(c, d, l'),

    with ``K`` the min-plus convolution of those mechanisms (each used at
    most once). ``W(l) = V_last(s_last, l)``. Validated against
    ``minsum_table`` (brute force) at m <= 2.
    """

    def __init__(self, slots, mechanisms):
        slots = np.asarray(slots, dtype=np.int64)
        self.num_detectors = len(slots)
        self.T = int(slots.max()) + 1
        self.dets = [np.flatnonzero(slots == t) for t in range(self.T)]
        self.n = [len(d) for d in self.dets]
        pos = np.empty(len(slots), dtype=np.int64)
        for d in self.dets:
            pos[d] = np.arange(len(d))
        kernels = []
        for t in range(self.T):
            n_prev = self.n[t - 1] if t else 0
            k = np.full(2 ** (n_prev + self.n[t] + 1), np.inf)
            k[0] = 0.0
            kernels.append(k)
        for dets, flip, w in mechanisms:
            if not dets:
                if flip:
                    t, v = 0, 1 << self.n[0]
                else:
                    continue
            else:
                ts = slots[list(dets)]
                t = int(ts.max())
                if int(ts.min()) < t - 1:
                    raise ValueError("mechanism spans non-adjacent slots")
                n_prev = self.n[t - 1] if t else 0
                v = 0
                for d in dets:
                    v ^= ((1 << (n_prev + int(pos[d]))) if slots[d] == t
                          else (1 << int(pos[d])))
                if flip:
                    v ^= 1 << (n_prev + self.n[t])
            idx = np.arange(kernels[t].size)
            kernels[t] = np.minimum(kernels[t], kernels[t][idx ^ v] + w)
        self.kernels = kernels

    def class_weights(self, events: np.ndarray) -> np.ndarray:
        events = np.asarray(events, dtype=bool)
        packed = np.packbits(events, axis=1)
        uniq, inverse = np.unique(packed, axis=0, return_inverse=True)
        ev = np.unpackbits(uniq, axis=1, count=self.num_detectors).astype(bool)
        S = ev.shape[0]
        rows = np.arange(S)
        sval = [ev[:, self.dets[t]].astype(np.int64)
                @ (1 << np.arange(self.n[t])).astype(np.int64)
                for t in range(self.T)]
        V = np.broadcast_to(self.kernels[0], (S, self.kernels[0].size)).copy()
        for t in range(1, self.T):
            n_prev, n_cur = self.n[t - 1], self.n[t]
            K = self.kernels[t].reshape(2, 2 ** n_cur, 2 ** n_prev)  # [l', d, c]
            Vp = V.reshape(S, 2, 2 ** n_prev)                      # [l, b]
            new = np.full((S, 2, 2 ** n_cur), np.inf)
            s = sval[t - 1]
            for c in range(2 ** n_prev):
                g = Vp[rows, :, s ^ c]                             # (S, 2) over l
                for l_out in (0, 1):
                    for lp in (0, 1):
                        cand = g[:, l_out ^ lp][:, None] + K[lp, :, c][None, :]
                        np.minimum(new[:, l_out, :], cand, out=new[:, l_out, :])
            V = new.reshape(S, -1)
        last = self.n[-1]
        s = sval[-1]
        W = np.stack([V[rows, s], V[rows, s + (1 << last)]], axis=1)
        return W[inverse.reshape(-1)]


def pad(events: np.ndarray, width: int) -> np.ndarray:
    out = np.zeros((events.shape[0], width), dtype=np.uint8)
    out[:, :events.shape[1]] = events
    return out


# ---------------------------------------------------------------------------
# H1: single-fault census
# ---------------------------------------------------------------------------

def census_task(args) -> dict:
    source, seq_index, sequence, depth, p = args
    lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
    und = lc.circuit.detector_error_model()
    dec = lc.circuit.detector_error_model(decompose_errors=True)
    mechs = undecomposed_mechanisms(und)
    keys = [(d, f) for d, f, _ in mechs]
    n = und.num_detectors
    k = len(mechs)
    syn = np.zeros((k, n), dtype=np.uint8)
    for i, (dets, _, _) in enumerate(mechs):
        syn[i, list(dets)] = 1
    flips = np.array([f for _, f, _ in mechs], dtype=bool)
    probs = np.array([q for _, _, q in mechs])
    pred: dict = {"none": np.zeros(k, dtype=bool)}
    pm = pymatching.Matching.from_detector_error_model(dec)
    pred["mwpm"] = pm.decode_batch(syn)[:, 0].astype(bool)
    stripped, nstrip = strip_detector_free(dec)
    pmc = pymatching.Matching.from_detector_error_model(
        stripped, enable_correlations=True)
    pred["mwpm_corr"] = pmc.decode_batch(
        syn, enable_correlations=True)[:, 0].astype(bool)
    mld = M.TransferMatrixMLD(und, M.lrb_detector_slots(lc))
    llr = mld.log_likelihood_ratio(syn.astype(bool))
    pred["mld"] = llr < 0
    ca = classaware_matching(dec, n)
    pred["classaware"] = ca.decode_batch(
        pad(syn, ca.num_detectors))[:, 0].astype(bool)
    # Exact minimum-weight error on the undecomposed hypergraph (own sweep);
    # ties are broken both ways to show they do not matter at first order.
    mwe_w = MinPlusSweep(M.lrb_detector_slots(lc),
                         [(d, f, weight(q)) for d, f, q in mechs]
                         ).class_weights(syn.astype(bool))
    pred["mwe"] = (mwe_w[:, 1] - mwe_w[:, 0]) < -TIE
    pred["mwe_tieflip"] = (mwe_w[:, 1] - mwe_w[:, 0]) < TIE
    classes: dict = {}
    for dets, flip, q in mechs:
        classes.setdefault(dets, [0.0, 0.0])[int(flip)] += q
    e1 = sum(min(a, b) for a, b in classes.values())
    row = {"source": source, "seq": seq_index, "depth": depth, "p": p,
           "num_detectors": n, "mechanisms": k,
           "unique_symptoms": int(len(set(keys)) == len(keys)),
           "undetectable_logical_mass": float(
               sum(q for d, f, q in mechs if not d and f) / p),
           "mass": float(probs.sum() / p), "E1": e1 / p,
           "mld_exact_ties": int((np.abs(llr) < TIE).sum()),
           "stripped_lines": nstrip}
    for name, prediction in pred.items():
        row["c_" + name] = float(probs[prediction != flips].sum() / p)
    # Decoders that disagree with the per-class argmax on single faults whose
    # two classes are not tied (first-order ties cost the same either way).
    wrong_vs_argmax = 0.0
    for i, (dets, flip, q) in enumerate(mechs):
        a, b = classes[dets]
        if abs(a - b) > 1e-3 * max(a, b) and pred["mld"][i] != (b > a):
            wrong_vs_argmax += q
    row["mld_not_argmax_mass"] = wrong_vs_argmax / p
    return row


def slope(depths, values) -> float:
    return float(np.polyfit(np.asarray(depths, float), np.asarray(values), 1)[0])


def summarize_census(rows: list[dict], out: Path) -> dict:
    decoders = ("mld", "E1", "mwe", "mwe_tieflip", "mwpm", "mwpm_corr",
                "classaware", "none")
    summary: dict = {}
    for source in sorted({r["source"] for r in rows}):
        for p in sorted({r["p"] for r in rows if r["source"] == source}):
            sub = [r for r in rows if r["source"] == source and r["p"] == p]
            seqs = sorted({r["seq"] for r in sub})
            depths = sorted({r["depth"] for r in sub})
            table = {(r["seq"], r["depth"]): r for r in sub}
            key = f"{source}_p{p:g}"
            entry: dict = {"num_seq": len(seqs), "depths": depths,
                           "all_unique_symptoms": all(r["unique_symptoms"]
                                                      for r in sub),
                           "max_undetectable_logical_mass": max(
                               r["undetectable_logical_mass"] for r in sub),
                           "max_mld_not_argmax_mass": max(
                               r["mld_not_argmax_mass"] for r in sub),
                           "max_abs_mld_minus_E1": max(
                               abs(r["c_mld"] - r["E1"]) for r in sub),
                           "max_abs_mwe_minus_E1_both_tie_rules": max(
                               max(abs(r["c_mwe"] - r["E1"]),
                                   abs(r["c_mwe_tieflip"] - r["E1"]))
                               for r in sub)}
            per_depth = {}
            for d in depths:
                vals = {name: np.array([table[(s, d)]["c_" + name]
                                        if name != "E1" else table[(s, d)]["E1"]
                                        for s in seqs]) for name in decoders}
                diff = vals["mwpm"] - vals["none"]
                per_depth[d] = {
                    **{name: [float(v.mean()), float(v.std(ddof=1) / math.sqrt(len(v)))
                              if len(v) > 1 else 0.0] for name, v in vals.items()},
                    "mwpm_minus_none": [float(diff.mean()),
                                        float(diff.std(ddof=1) / math.sqrt(len(diff)))
                                        if len(diff) > 1 else 0.0],
                    "frac_seq_mwpm_worse_than_none": float((diff > 0).mean()),
                    "frac_seq_corr_worse_than_plain": float(
                        (vals["mwpm_corr"] > vals["mwpm"] + 1e-9).mean())}
            entry["per_depth"] = per_depth
            fits = {}
            for label, dset in (("B_depths_0_2_6_14_22", B_DEPTHS),
                                ("depths_2_6_14_22", (2, 6, 14, 22)),
                                ("all_available_depths", tuple(depths)),
                                ("all_depths_ge_2", tuple(d for d in depths
                                                          if d >= 2))):
                if not all(d in depths for d in dset) or len(dset) < 2:
                    continue
                fit = {}
                for name in decoders:
                    col = "E1" if name == "E1" else "c_" + name
                    s = np.array([slope(dset, [table[(q, d)][col] for d in dset])
                                  for q in seqs])
                    fit[name] = [float(s.mean()), float(s.std(ddof=1) / math.sqrt(len(s)))
                                 if len(s) > 1 else 0.0]
                fits[label] = fit
            entry["slopes_per_depth_step"] = fits
            summary[key] = entry
    with open(out / "census_rows.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Cross-check against B's first_order.csv on B's own sequences.
    b_rows = {}
    with open(B_RESULTS / "first_order.csv") as handle:
        for r in csv.DictReader(handle):
            b_rows[(int(r["seq"]), int(r["depth"]), float(r["p"]))] = r
    maxdiff: dict = {}
    for r in rows:
        if r["source"] != "B":
            continue
        b = b_rows.get((r["seq"], r["depth"], r["p"]))
        if b is None:
            continue
        for mine, theirs in (("mld", "c_mld"), ("mwpm", "c_mwpm"),
                             ("mwpm_corr", "c_mwpm_corr"), ("none", "c_none"),
                             ("classaware", "c_mwpm_classaware"),
                             ("E1", "c_mwe"), ("mwe", "c_mwe")):
            value = r["E1"] if mine == "E1" else r["c_" + mine]
            delta = abs(value - float(b[theirs]))
            maxdiff[mine] = max(maxdiff.get(mine, 0.0), delta)
    summary["max_abs_diff_vs_B_first_order_csv"] = maxdiff
    return summary


# ---------------------------------------------------------------------------
# H4: detector-free decomposition components
# ---------------------------------------------------------------------------

def _graph_signature(matching) -> dict:
    sig = {}
    for u, v, data in matching.edges():
        key = (int(u), None if v is None else int(v))
        sig[key] = (tuple(sorted(data["fault_ids"])), float(data["weight"]))
    return sig


def _same_graph(a, b) -> bool:
    sa, sb = _graph_signature(a), _graph_signature(b)
    if set(sa) != set(sb):
        return False
    return all(sa[k][0] == sb[k][0] and math.isclose(sa[k][1], sb[k][1],
                                                      rel_tol=1e-12)
               for k in sa)


def h4_task(args) -> dict:
    source, seq_index, sequence, depth, p = args
    lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
    dec = lc.circuit.detector_error_model(decompose_errors=True)
    lines = list(error_lines(dec))
    free_lines = [(q, comps) for q, comps in lines if any(not d for d, _ in comps)]
    kinds = Counter()
    misattributed = 0
    for q, comps in free_lines:
        for d, f in comps:
            if not d:
                kinds["L0" if f else "empty"] += 1
        true_flip = sum(f for _, f in comps) % 2
        kept_flip = sum(f for d, f in comps if d) % 2
        misattributed += int(true_flip != kept_flip)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plain = pymatching.Matching.from_detector_error_model(dec)
    stripped, changed = strip_detector_free(dec)
    plain_stripped = pymatching.Matching.from_detector_error_model(stripped)
    corr_error = ""
    corr_built = False
    try:
        pymatching.Matching.from_detector_error_model(dec,
                                                      enable_correlations=True)
        corr_built = True
    except ValueError as exc:
        corr_error = "ValueError: " + str(exc).splitlines()[0][:160]
    except Exception as exc:  # report anything else verbatim
        corr_error = f"{type(exc).__name__}: " + str(exc).splitlines()[0][:160]
    corr_stripped_ok = True
    try:
        pymatching.Matching.from_detector_error_model(stripped,
                                                      enable_correlations=True)
    except Exception:
        corr_stripped_ok = False
    return {"source": source, "seq": seq_index, "depth": depth, "p": p,
            "decomposed_lines": len(lines), "free_lines": len(free_lines),
            "free_mass_over_p": sum(q for q, _ in free_lines) / p,
            "free_components_L0": kinds["L0"],
            "free_components_empty": kinds["empty"],
            "lines_whose_kept_parts_carry_wrong_L0": misattributed,
            "plain_warnings": len(caught),
            "plain_graph_equals_stripped_graph": int(_same_graph(plain,
                                                                 plain_stripped)),
            "stripped_changed": changed,
            "corr_raw_built": int(corr_built), "corr_raw_error": corr_error,
            "corr_stripped_built": int(corr_stripped_ok)}


def h4_minimal() -> dict:
    """Hand-written DEM: mechanism D0 D1 without L0, decomposed with a remainder."""
    text = "error(0.1) D0 L0 ^ D1 ^ L0\nerror(0.01) D0\nerror(0.01) D1"
    dem = stim.DetectorErrorModel(text)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        plain = pymatching.Matching.from_detector_error_model(dem)
    edges = {(int(u), v): sorted(d["fault_ids"]) for u, v, d in plain.edges()}
    prediction = int(plain.decode(np.array([1, 1], dtype=np.uint8))[0])
    try:
        pymatching.Matching.from_detector_error_model(dem,
                                                      enable_correlations=True)
        corr = "built"
    except ValueError as exc:
        corr = "ValueError: " + str(exc).splitlines()[0][:200]
    return {"dem": text, "plain_warnings": len(caught),
            "plain_edges_fault_ids": {str(k): v for k, v in edges.items()},
            "plain_prediction_for_D0_D1": prediction,
            "true_flip_of_D0_D1_mechanism": 0, "correlated": corr}


def h4_example_line(sequence) -> list[str]:
    """B's quoted example: m = 2, sequence 0, p = 2e-3."""
    lc = Q.build_lrb_circuit(sequence, 2, 2e-3, terminal="data5")
    dec = lc.circuit.detector_error_model(decompose_errors=True)
    out = []
    for ins in dec.flattened():
        if ins.type == "error" and any(not d for d, _ in components(ins)):
            out.append(str(ins))
    return out


def summarize_h4(rows: list[dict]) -> dict:
    summary: dict = {}
    for source in sorted({r["source"] for r in rows}):
        for depth in sorted({r["depth"] for r in rows}):
            sub = [r for r in rows if r["source"] == source and r["depth"] == depth]
            if not sub:
                continue
            mass = np.array([r["free_mass_over_p"] for r in sub])
            nl = np.array([r["free_lines"] for r in sub])
            has = [r for r in sub if r["free_lines"] > 0]
            summary[f"{source}_m{depth}"] = {
                "num_seq": len(sub),
                "free_lines_mean_min_max": [float(nl.mean()), int(nl.min()),
                                            int(nl.max())],
                "free_mass_over_p_mean_min_max": [float(mass.mean()),
                                                  float(mass.min()),
                                                  float(mass.max())],
                "empty_components": int(sum(r["free_components_empty"]
                                            for r in sub)),
                "misattributed_lines_total": int(sum(
                    r["lines_whose_kept_parts_carry_wrong_L0"] for r in sub)),
                "free_lines_total": int(nl.sum()),
                "plain_warnings_total": int(sum(r["plain_warnings"] for r in sub)),
                "plain_graph_equals_stripped_all": all(
                    r["plain_graph_equals_stripped_graph"] for r in sub),
                "corr_raises_ValueError_iff_free_lines": all(
                    (r["corr_raw_built"] == 0) == (r["free_lines"] > 0)
                    for r in sub),
                "corr_errors_seen": sorted({r["corr_raw_error"][:80]
                                            for r in has}),
                "corr_on_stripped_builds_all": all(r["corr_stripped_built"]
                                                   for r in sub)}
    return summary


# ---------------------------------------------------------------------------
# H6: physical RB per depth step
# ---------------------------------------------------------------------------

def h6(new_sequences) -> dict:
    catalog = C.SINGLE_QUBIT_CLIFFORDS
    fwd = [len(C.clifford_gate_sequence(d)) for d in catalog]
    inv = [len(C.clifford_inverse_sequence(d)) for d in catalog]
    tokens_per_step = [a + b for a, b in zip(fwd, inv)]
    mean_tokens = float(np.mean(fwd))
    first_order = float(np.mean(tokens_per_step)) * 0.1 * 2.0 / 3.0
    exact = {}
    for p_rb in (1e-5, 1e-4, 1e-3, 5e-3, 1e-2, 2.5e-2, 5e-2):
        lam = 1.0 - 4.0 * 0.1 * p_rb / 3.0
        f = float(np.mean([lam ** t for t in tokens_per_step]))
        exact[f"{p_rb:g}"] = (1.0 - f) / 2.0 / p_rb
    # Stim DEM check: exact flip probability vs (1 - lam^N)/2.
    worst = 0.0
    checked = 0
    for sequence in new_sequences:
        for depth in (1, 6, 22):
            for p_rb in (5e-3, 5e-2):
                circuit = Q.build_physical_rb_circuit(sequence, depth, p_rb)
                q_flip = 0.0
                for q, comps in error_lines(circuit.detector_error_model()):
                    if sum(f for _, f in comps) % 2:
                        q_flip = q_flip * (1 - q) + q * (1 - q_flip)
                tokens = sum(len(C.clifford_gate_sequence(d))
                             + len(C.clifford_inverse_sequence(d))
                             for d in sequence[:depth])
                lam = 1.0 - 4.0 * 0.1 * p_rb / 3.0
                formula = (1.0 - lam ** tokens) / 2.0
                worst = max(worst, abs(q_flip - formula) / formula)
                checked += 1
    return {"catalog_size": len(catalog), "distinct": len(set(catalog)),
            "tokens_forward": dict(zip(catalog, fwd)),
            "tokens_inverse_equal_forward": fwd == inv,
            "total_tokens_catalog": int(sum(fwd)),
            "mean_tokens_per_clifford": mean_tokens,
            "first_order_error_per_step_over_p_rb": first_order,
            "exact_seqavg_error_per_step_over_p_rb": exact,
            "stim_dem_vs_formula_max_rel_diff": worst,
            "stim_dem_checks": checked}


# ---------------------------------------------------------------------------
# Sampled spot checks (H2; H3 and H5 at m <= 2)
# ---------------------------------------------------------------------------

def _hist(gap, wrong, edges) -> np.ndarray:
    bins = np.clip(np.searchsorted(edges, gap, side="right") - 1, 0,
                   len(edges) - 2)
    out = np.zeros((len(edges) - 1, 2), dtype=np.int64)
    np.add.at(out, (bins, wrong.astype(np.int64)), 1)
    return out


class _TableLookup:
    """Brute-force table with the same ``class_weights`` interface as the sweep."""

    def __init__(self, n, mechanisms):
        self.n = n
        self.table = minsum_table(n, mechanisms)
        self.powers = (1 << np.arange(n)).astype(np.int64)

    def class_weights(self, events):
        index = np.asarray(events, dtype=np.int64) @ self.powers
        return np.stack([self.table[index], self.table[index + (1 << self.n)]],
                        axis=1)


def sample_task(args) -> dict:
    label, p, seq_index, sequence, depth, shots, seed = args[:7]
    use_sweep = len(args) > 7 and bool(args[7])
    start = time.perf_counter()
    lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
    und = lc.circuit.detector_error_model()
    dec = lc.circuit.detector_error_model(decompose_errors=True)
    n = und.num_detectors
    pm = pymatching.Matching.from_detector_error_model(dec)
    slots = M.lrb_detector_slots(lc)
    mld = M.TransferMatrixMLD(und, slots)
    tables = {}
    mech_sets = {
        "mwe": [(d, f, weight(q)) for d, f, q in undecomposed_mechanisms(und)],
        "classaware": [(d, f, weight(q)) for d, f, q in classaware_edges(dec)],
        "mwpm_own": pymatching_graph_edges(pm)}
    if n <= 18 and not use_sweep:
        tables = {k: _TableLookup(n, v) for k, v in mech_sets.items()}
    elif use_sweep:
        tables = {k: MinPlusSweep(slots, v) for k, v in mech_sets.items()}
    # Tie-rule variants (ties -> "flip" instead of "no flip").
    tie_variants = [k + "_tieflip" for k in ("mwe", "classaware")
                    if k in tables]
    decoders = ["mld", "mwpm", "none"] + list(tables) + tie_variants
    soft = ["mld"] + list(tables)
    res = {"label": label, "p": p, "seq": seq_index, "depth": depth,
           "seed": seed, "num_detectors": n, "shots": 0,
           "err": {d: 0 for d in decoders},
           "both_wrong_with_mld": {d: 0 for d in decoders},
           "hist": {d: np.zeros((len(GAP_EDGES) - 1, 2), np.int64) for d in soft},
           "calib": {d: np.zeros((len(CALIB_EDGES) - 1, 3)) for d in soft},
           "zero": [0, 0], "pm_vs_own_disagree": 0, "pm_vs_own_decided": 0,
           "own_weight_vs_pm_maxdiff": 0.0, "ties": {k: 0 for k in tables},
           "use_sweep": use_sweep}
    sampler = lc.circuit.compile_detector_sampler(seed=seed)
    done = 0
    while done < shots:
        nb = min(100_000, shots - done)
        dets, obs = sampler.sample(nb, separate_observables=True)
        flip = obs[:, 0]
        pred, gaps = {}, {}
        llr = mld.log_likelihood_ratio(dets)
        pred["mld"], gaps["mld"] = llr < 0, np.abs(llr)
        if "mwpm_own" in tables:
            pm_pred, pm_w = pm.decode_batch(dets.astype(np.uint8),
                                            return_weights=True)
        else:
            pm_pred = pm.decode_batch(dets.astype(np.uint8))
        pred["mwpm"] = pm_pred[:, 0].astype(bool)
        pred["none"] = np.zeros(nb, dtype=bool)
        if tables:
            for name, decoder in tables.items():
                weights = decoder.class_weights(dets)
                w0, w1 = weights[:, 0], weights[:, 1]
                pred[name] = (w1 - w0) < -TIE
                gaps[name] = np.abs(w1 - w0)
                res["ties"][name] += int((gaps[name] < TIE).sum())
                if name + "_tieflip" in decoders:
                    pred[name + "_tieflip"] = (w1 - w0) < TIE
                if name == "mwpm_own":
                    decided = gaps[name] > 1e-4
                    res["pm_vs_own_decided"] += int(decided.sum())
                    res["pm_vs_own_disagree"] += int(
                        (pred["mwpm"] != pred[name])[decided].sum())
                    res["own_weight_vs_pm_maxdiff"] = max(
                        res["own_weight_vs_pm_maxdiff"],
                        float(np.max(np.abs(np.minimum(w0, w1) - pm_w))))
        mld_wrong = pred["mld"] != flip
        for name in decoders:
            wrong = pred[name] != flip
            res["err"][name] += int(wrong.sum())
            res["both_wrong_with_mld"][name] += int((wrong & mld_wrong).sum())
        for name in soft:
            wrong = pred[name] != flip
            res["hist"][name] += _hist(gaps[name], wrong, GAP_EDGES)
            bins = np.clip(np.searchsorted(CALIB_EDGES, gaps[name],
                                           side="right") - 1, 0,
                           len(CALIB_EDGES) - 2)
            predicted = 1.0 / (1.0 + np.exp(np.minimum(gaps[name], 700.0)))
            np.add.at(res["calib"][name], (bins, 0), 1.0)
            np.add.at(res["calib"][name], (bins, 1), wrong.astype(float))
            np.add.at(res["calib"][name], (bins, 2), predicted)
        clean = ~dets.any(axis=1)
        res["zero"][0] += int(clean.sum())
        res["zero"][1] += int((clean & flip).sum())
        res["shots"] += nb
        done += nb
    res["seconds"] = time.perf_counter() - start
    return res


def error_at_acceptance(hist: np.ndarray, target: float) -> float:
    """Accepted error when the top ``target`` fraction by gap is kept."""
    counts = hist.sum(axis=1)[::-1].astype(float)
    wrong = hist[:, 1][::-1].astype(float)
    total = counts.sum()
    need = target * total
    cum = np.cumsum(counts)
    k = min(int(np.searchsorted(cum, need - 1e-9)), len(cum) - 1)
    before = cum[k - 1] if k > 0 else 0.0
    frac = 0.0 if counts[k] == 0 else (need - before) / counts[k]
    return float((wrong[:k].sum() + frac * wrong[k]) / need)


def jackknife(per_seq: np.ndarray, stat) -> tuple[float, float]:
    total = per_seq.sum(axis=0)
    full = stat(total)
    k = per_seq.shape[0]
    if k < 2:
        return float(full), float("nan")
    loo = np.array([stat(total - per_seq[i]) for i in range(k)])
    return float(full), float(np.sqrt((k - 1) / k * np.sum((loo - loo.mean()) ** 2)))


def _load_b_ler() -> dict:
    out = {}
    with open(B_RESULTS / "ler.csv") as handle:
        for r in csv.DictReader(handle):
            out[(float(r["p"]), int(r["depth"]), r["decoder"])] = r
    return out


def _load_b_calibration() -> dict:
    out = {}
    with open(B_RESULTS / "calibration.csv") as handle:
        for r in csv.DictReader(handle):
            out[(float(r["p"]), int(r["depth"]), r["decoder"],
                 float(r["gap_lo"]))] = r
    return out


def _load_b_acceptance() -> dict:
    out = {}
    with open(B_RESULTS / "acceptance.csv") as handle:
        for r in csv.DictReader(handle):
            out[(float(r["p"]), int(r["depth"]), r["decoder"], r["target"])] = r
    return out


def _load_b_ladder() -> dict:
    err: dict = {}
    shots: dict = {}
    with open(B_RESULTS / "per_circuit.csv") as handle:
        for r in csv.DictReader(handle):
            key = (float(r["p"]), int(r["depth"]), r["decoder"])
            err[key] = err.get(key, 0) + int(r["errors"])
            shots[key] = shots.get(key, 0) + int(r["shots"])
    return {k: err[k] / shots[k] for k in err}


B_NAME = {"mld": "mld", "mwpm": "mwpm", "none": "none", "mwe": "mwe",
          "classaware": "mwpm_classaware", "mwpm_own": "mwpm_gap",
          "mwe_tieflip": "-", "classaware_tieflip": "-"}


def summarize_samples(results: list[dict]) -> dict:
    b_ler, b_cal = _load_b_ler(), _load_b_calibration()
    b_acc, b_lad = _load_b_acceptance(), _load_b_ladder()
    summary: dict = {}
    for label in sorted({r["label"] for r in results}):
        parts = sorted([r for r in results if r["label"] == label],
                       key=lambda r: r["seq"])
        p, depth = parts[0]["p"], parts[0]["depth"]
        decoders = list(parts[0]["err"])
        shots = np.array([r["shots"] for r in parts], float)
        total = shots.sum()
        mld_err = np.array([r["err"]["mld"] for r in parts], float)
        entry: dict = {"p": p, "depth": depth, "num_seq": len(parts),
                       "shots_total": int(total),
                       "seeds": [r["seed"] for r in parts],
                       "cpu_seconds": float(sum(r["seconds"] for r in parts)),
                       "decoders": {}}
        same_sequences = label.startswith("B_")
        for name in decoders:
            err = np.array([r["err"][name] for r in parts], float)
            both = np.array([r["both_wrong_with_mld"][name] for r in parts], float)
            ler, ler_jk = jackknife(np.stack([err, shots], 1),
                                    lambda s: s[0] / s[1])
            ratio, ratio_jk = jackknife(np.stack([err, mld_err], 1),
                                        lambda s: s[0] / s[1])
            # Shot-noise-only SE of the ratio (paired multinomial, delta method).
            ed, em, eb = err.sum(), mld_err.sum(), both.sum()
            pd, pmw, pb = ed / total, em / total, eb / total
            if name == "mld":
                ratio_shot = 0.0
            else:
                var = (pd * (1 - pd) / (ed / total) ** 2
                       + pmw * (1 - pmw) / (em / total) ** 2
                       - 2 * (pb - pd * pmw) / ((ed / total) * (em / total)))
                ratio_shot = float(ratio * math.sqrt(max(var, 0.0) / total))
            row = {"ler": ler, "ler_se_jk": ler_jk,
                   "ler_se_binom": math.sqrt(ler * (1 - ler) / total),
                   "ratio_to_mld": ratio, "ratio_se_jk": ratio_jk,
                   "ratio_se_shot": ratio_shot}
            b = b_ler.get((p, depth, B_NAME[name]))
            if b is not None:
                row["B_ler"] = float(b["ler"])
                row["B_ratio_to_mld"] = float(b["ratio_to_mld"])
                row["B_ratio_se_jk"] = float(b["ratio_se_jk"])
                if same_sequences:
                    # Same circuits, independent shots: compare at shot noise.
                    se = math.sqrt(2.0) * row["ler_se_binom"]
                    row["z_ler_vs_B_shot_noise"] = (ler - float(b["ler"])) / se
                    if name != "mld" and ratio_shot > 0:
                        row["z_ratio_vs_B_shot_noise"] = (
                            (ratio - float(b["ratio_to_mld"]))
                            / (math.sqrt(2.0) * ratio_shot))
                else:
                    # New sequences: compare with B's jackknife (sequence) SE.
                    se = math.sqrt(float(b["ratio_se_jk"]) ** 2
                                   + (ratio_jk if ratio_jk == ratio_jk else 0) ** 2)
                    if name != "mld" and se > 0:
                        row["z_ratio_vs_B_seq_spread"] = (
                            (ratio - float(b["ratio_to_mld"])) / se)
            entry["decoders"][name] = row
        if "mwe" in decoders:
            ler = {d: entry["decoders"][d]["ler"] for d in decoders}
            per_seq = {d: np.array([r["err"][d] for r in parts], float)
                       for d in decoders}

            def rung(a, b_):
                v = np.stack([per_seq[a] - per_seq[b_], shots], 1)
                return jackknife(v, lambda s: s[0] / s[1])

            ladder = {"merge_rule(mwpm-classaware)": rung("mwpm", "classaware"),
                      "decomposition(classaware-mwe)": rung("classaware", "mwe"),
                      "degeneracy_ties(mwe-mld)": rung("mwe", "mld"),
                      "total(mwpm-mld)": rung("mwpm", "mld"),
                      "mwpm-none": rung("mwpm", "none")}
            tot = ler["mwpm"] - ler["mld"]
            entry["ladder_1e-3"] = {k: [1e3 * v[0], 1e3 * v[1]]
                                    for k, v in ladder.items()}
            entry["ladder_shares"] = {
                "merge": (ler["mwpm"] - ler["classaware"]) / tot,
                "decomposition": (ler["classaware"] - ler["mwe"]) / tot,
                "degeneracy_ties": (ler["mwe"] - ler["mld"]) / tot}
            if "classaware_tieflip" in ler:
                entry["ladder_shares_ties_to_flip"] = {
                    "merge": (ler["mwpm"] - ler["classaware_tieflip"]) / tot,
                    "decomposition": (ler["classaware_tieflip"]
                                      - ler["mwe_tieflip"]) / tot,
                    "degeneracy_ties": (ler["mwe_tieflip"] - ler["mld"]) / tot}
            entry["tie_fraction"] = {
                k: sum(r["ties"][k] for r in parts) / total
                for k in parts[0]["ties"]}
            # Merge-rule share per sequence (sequence-to-sequence spread).
            entry["merge_share_per_seq"] = [
                float((r["err"]["mwpm"] - r["err"]["classaware"])
                      / (r["err"]["mwpm"] - r["err"]["mld"]))
                if r["err"]["mwpm"] != r["err"]["mld"] else float("nan")
                for r in parts]
            if (p, depth, "mwpm") in b_lad and same_sequences:
                bt = b_lad[(p, depth, "mwpm")] - b_lad[(p, depth, "mld")]
                entry["B_ladder_shares"] = {
                    "merge": (b_lad[(p, depth, "mwpm")]
                              - b_lad[(p, depth, "mwpm_classaware")]) / bt,
                    "decomposition": (b_lad[(p, depth, "mwpm_classaware")]
                                      - b_lad[(p, depth, "mwe")]) / bt,
                    "degeneracy_ties": (b_lad[(p, depth, "mwe")]
                                        - b_lad[(p, depth, "mld")]) / bt,
                    "total_1e-3": 1e3 * bt}
            entry["pm_vs_own_graph_disagreements"] = int(
                sum(r["pm_vs_own_disagree"] for r in parts))
            entry["pm_vs_own_graph_decided_shots"] = int(
                sum(r["pm_vs_own_decided"] for r in parts))
            entry["own_weight_vs_pm_maxdiff"] = float(
                max(r["own_weight_vs_pm_maxdiff"] for r in parts))
        # Calibration and acceptance at the zero-syndrome acceptance A0.
        soft = list(parts[0]["hist"])
        zs = np.array([r["zero"] for r in parts], float)
        a0 = zs[:, 0].sum() / total
        entry["A0"] = a0
        entry["zero_syndrome_error"] = zs[:, 1].sum() / zs[:, 0].sum()
        b0 = b_acc.get((p, depth, "detect_all_zero_syndrome", "zero_syndrome"))
        if b0 is not None:
            entry["B_A0"] = float(b0["acceptance"])
            entry["B_zero_syndrome_error"] = float(b0["accepted_error"])
        acc, cal = {}, {}
        for name in soft:
            hists = np.stack([r["hist"][name] for r in parts])
            flat = hists.reshape(len(parts), -1)
            for tlabel, level in (("A0", a0), ("0.9", 0.9)):
                value, se = jackknife(
                    flat, lambda s, level=level: error_at_acceptance(
                        s.reshape(-1, 2), level))
                row = {"accepted_error": value, "se_jk": se}
                bt = "zero_syndrome" if tlabel == "A0" else "0.9"
                b = b_acc.get((p, depth, B_NAME[name], bt))
                if b is not None:
                    row["B_accepted_error"] = float(b["accepted_error"])
                acc[f"{name}@{tlabel}"] = row
            cal_sum = sum(r["calib"][name] for r in parts)
            rows = []
            for k in range(len(CALIB_EDGES) - 1):
                if cal_sum[k, 0] == 0:
                    continue
                emp = cal_sum[k, 1] / cal_sum[k, 0]
                row = {"bin": [float(CALIB_EDGES[k]), float(CALIB_EDGES[k + 1])],
                       "shots": int(cal_sum[k, 0]), "empirical": emp,
                       "empirical_se": math.sqrt(emp * (1 - emp) / cal_sum[k, 0]),
                       "predicted": cal_sum[k, 2] / cal_sum[k, 0]}
                b = b_cal.get((p, depth, B_NAME[name], float(CALIB_EDGES[k])))
                if b is not None:
                    row["B_empirical"] = float(b["empirical_error"])
                    row["B_predicted"] = float(b["predicted_error_from_gap"])
                rows.append(row)
            cal[name] = rows
        entry["acceptance"] = acc
        entry["calibration"] = cal
        summary[label] = entry
    return summary


SAMPLE_POINTS = (
    # label, p, depth, sequence source, number of sequences
    ("B_p0.0005_m2", 5e-4, 2, "B", 10),
    ("B_p0.0005_m14", 5e-4, 14, "B", 10),
    ("B_p0.01_m2", 1e-2, 2, "B", 10),
    ("B_p0.01_m0", 1e-2, 0, "B", 10),
    ("B_p0.002_m6", 2e-3, 6, "B", 10),
    ("new_p0.0005_m6", 5e-4, 6, "new", 10),
    ("new_p0.002_m2", 2e-3, 2, "new", 10),
    # Below B's grid: does MWPM/MLD stay <= 1.79 wherever LER_MLD < 0.4?
    ("B_p0.0001_m6", 1e-4, 6, "B", 10),
    ("B_p0.0001_m14", 1e-4, 14, "B", 10),
)


SWEEP_POINTS = (
    # Same as SAMPLE_POINTS, decoded with the own min-sum sweep (m > 2).
    ("SW_B_p0.0005_m6", 5e-4, 6, "B", 10),
    ("SW_B_p0.002_m6", 2e-3, 6, "B", 10),
    ("SW_B_p0.0005_m22", 5e-4, 22, "B", 10),
    ("SW_new_p0.0005_m6", 5e-4, 6, "new", 10),
)
SWEEP_LABEL_OFFSET = 20


def validate_sweep(b_seqs) -> dict:
    """MinPlusSweep vs brute force (m <= 2) and vs PyMatching weights (m > 2)."""
    out = {}
    for depth in (0, 1, 2):
        for p in (2e-3, 3e-2):
            lc = Q.build_lrb_circuit(b_seqs[1], depth, p, terminal="data5")
            und = lc.circuit.detector_error_model()
            dec = lc.circuit.detector_error_model(decompose_errors=True)
            n = und.num_detectors
            slots = M.lrb_detector_slots(lc)
            dets, _ = lc.circuit.compile_detector_sampler(seed=99).sample(
                3000, separate_observables=True)
            for name, mechs in (
                    ("mwe", [(d, f, weight(q)) for d, f, q
                             in undecomposed_mechanisms(und)]),
                    ("classaware", [(d, f, weight(q)) for d, f, q
                                    in classaware_edges(dec)]),
                    ("pymatching", pymatching_graph_edges(
                        pymatching.Matching.from_detector_error_model(dec)))):
                a = MinPlusSweep(slots, mechs).class_weights(dets)
                b = _TableLookup(n, mechs).class_weights(dets)
                finite = np.isfinite(b)
                same_inf = bool(np.array_equal(np.isfinite(a), finite))
                diff = float(np.max(np.abs(a[finite] - b[finite]))) if finite.any() else 0.0
                out[f"brute_m{depth}_p{p:g}_{name}"] = [diff, same_inf]
    for depth in (6, 14, 22):
        lc = Q.build_lrb_circuit(b_seqs[0], depth, 5e-3, terminal="data5")
        dec = lc.circuit.detector_error_model(decompose_errors=True)
        pm = pymatching.Matching.from_detector_error_model(dec)
        dets, _ = lc.circuit.compile_detector_sampler(seed=98).sample(
            5000, separate_observables=True)
        pred, w = pm.decode_batch(dets.astype(np.uint8), return_weights=True)
        cw = MinPlusSweep(M.lrb_detector_slots(lc),
                          pymatching_graph_edges(pm)).class_weights(dets)
        decided = np.abs(cw[:, 1] - cw[:, 0]) > 1e-4
        out[f"pymatching_weight_m{depth}"] = [
            float(np.max(np.abs(cw.min(axis=1) - w))),
            int(((cw[:, 1] < cw[:, 0]) != pred[:, 0].astype(bool))[decided].sum())]
    return out


def sample_seed(label_index: int, seq: int, depth: int) -> int:
    """Fresh seeds, disjoint from B's (202610282 + 1e6*round(1e5 p) + ...)."""
    return SAMPLE_SEED + 100_000_000 * label_index + 1000 * seq + depth


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("what", choices=("census", "h4", "h6", "sample", "all",
                                         "validate", "sweep"))
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--num-new", type=int, default=20)
    parser.add_argument("--census-p", default="1e-5,1e-6")
    parser.add_argument("--shots", type=int, default=1_000_000)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    b_seqs = E.clifford_sequences(10, 22, B_SEQ_SEED)
    new_seqs = E.clifford_sequences(args.num_new, 22, NEW_SEQ_SEED)
    assert not any(s in b_seqs for s in new_seqs)
    report: dict = {"versions": {"stim": stim.__version__,
                                 "pymatching": pymatching.__version__},
                    "new_seq_seed": NEW_SEQ_SEED, "num_new": args.num_new,
                    "new_sequences": new_seqs}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        if args.what in ("census", "all"):
            p_list = [float(x) for x in args.census_p.split(",")]
            tasks = [("new", i, s, d, p) for p in p_list
                     for i, s in enumerate(new_seqs) for d in ALL_DEPTHS]
            tasks += [("B", i, s, d, p) for p in p_list
                      for i, s in enumerate(b_seqs) for d in ALL_DEPTHS]
            tasks.sort(key=lambda t: -t[3])
            rows = list(pool.map(census_task, tasks, chunksize=4))
            report["census"] = summarize_census(rows, out)
            print(f"[{time.perf_counter() - start:7.1f}s] census done "
                  f"({len(rows)} circuits)", flush=True)
        if args.what in ("h4", "all"):
            tasks = [("new", i, s, d, 2e-3) for i, s in enumerate(new_seqs)
                     for d in (2, 6, 14, 22)]
            tasks += [("B", i, s, d, 2e-3) for i, s in enumerate(b_seqs)
                      for d in (2, 6, 14, 22)]
            rows = list(pool.map(h4_task, tasks))
            with open(out / "h4_rows.csv", "w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            report["h4"] = {"minimal_dem": h4_minimal(),
                            "B_example_m2_seq0_p2e-3": h4_example_line(b_seqs[0]),
                            "summary": summarize_h4(rows)}
            # Linearity in p of the free-component mass (p = 1e-5 vs 2e-3).
            lin = [h4_task(("new", i, new_seqs[i], 22, 1e-5))["free_mass_over_p"]
                   for i in range(3)]
            report["h4"]["free_mass_over_p_at_1e-5_new_m22_seq0_2"] = lin
            print(f"[{time.perf_counter() - start:7.1f}s] h4 done", flush=True)
        if args.what in ("h6", "all"):
            report["h6"] = h6(new_seqs[:5])
            print(f"[{time.perf_counter() - start:7.1f}s] h6 done", flush=True)
        if args.what in ("sample", "all"):
            tasks = []
            for li, (label, p, depth, source, nseq) in enumerate(SAMPLE_POINTS):
                seqs = b_seqs if source == "B" else new_seqs
                for si in range(nseq):
                    tasks.append((label, p, si, seqs[si], depth, args.shots,
                                  sample_seed(li, si, depth)))
            tasks.sort(key=lambda t: -t[4])
            results = list(pool.map(sample_task, tasks))
            report["sample"] = summarize_samples(results)
            print(f"[{time.perf_counter() - start:7.1f}s] sampling done",
                  flush=True)
        if args.what == "validate":
            report["validate_sweep"] = validate_sweep(b_seqs)
            print(f"[{time.perf_counter() - start:7.1f}s] validation done",
                  flush=True)
        if args.what == "sweep":
            tasks = []
            for li, (label, p, depth, source, nseq) in enumerate(SWEEP_POINTS):
                seqs = b_seqs if source == "B" else new_seqs
                for si in range(nseq):
                    tasks.append((label, p, si, seqs[si], depth, args.shots,
                                  sample_seed(SWEEP_LABEL_OFFSET + li, si,
                                              depth), True))
            tasks.sort(key=lambda t: -t[4])
            results = list(pool.map(sample_task, tasks))
            report["sweep"] = summarize_samples(
                [dict(r, label=r["label"][3:]) for r in results])
            print(f"[{time.perf_counter() - start:7.1f}s] sweep sampling done",
                  flush=True)
    name = "verify_report.json" if args.what == "all" else f"verify_{args.what}.json"
    (out / name).write_text(json.dumps(_jsonable(report), indent=1))
    print(f"wrote {out / name} in {time.perf_counter() - start:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
