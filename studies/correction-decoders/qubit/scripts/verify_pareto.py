"""Adversarial verification of work package E (soft-output Pareto vs LRB-D).

Independent of E's code: ``pareto.py`` and ``hybrid_rules.py`` are never
imported, and ``lrb_qubit.acceptance_rules`` only inside ``c13`` and ``c14``
to compare E's ``tl_d5`` rule with ours on identical inputs (never to compute
a verdict). Everything else is built from Stim 1.15 and the lead's model
modules (``circuits``, ``baseline``, ``mld``, and ``exact`` in the ``exact``
subcommand). E's result CSVs are read only to compare numbers.

Subcommands (outputs go to ``--out``, default ``results/E-verify``):

``c4``    unif=1 / unif=1 + final are the gap rule at their acceptance.
          (a) Fresh samples, p = 2e-3, m in {2, 6}, sequences 0-2: exact LLR
          (lead's ``TransferMatrixMLD``) with all detectors (mld5) and with the
          check rounds only (mld2, data-2 DEM). Checks that the all-zero
          syndrome has a strictly larger |LLR| than every other observed
          syndrome and that the threshold just below it accepts exactly the
          LRB-D shot set (lead's ``raw_syndromes`` + ``accept_mask``).
          (b) Exhaustive: every syndrome at m = 2 (own brute-force joint
          distribution, independent of the lead's decoder) and m = 4 (lead's
          decoder on all 2^22 / 2^20 syndromes).
          (c) Every syndrome produced by one or two DEM mechanisms at m = 6
          and m = 22, p in {2e-3, 1e-2}.
          Also checks that the data-2 DEM is the exact marginal of the data-5
          DEM (m = 2 brute force), i.e. that mld2 is the exact check-round
          posterior.

``c5c10`` C5 (data-2 cap) and C10 (fixed-threshold bound) on exhaustive
          syndrome tables (m = 2, 4) and E's grid CSVs.

``c12``   Pauli algebra with ``stim.PauliString`` and fault injection into
          noiseless circuits after every SWAP(2,3) / CZ(2,3).

``c13``   Own implementation of ``tl_d5`` from the definition in the E note,
          derived from the detector definitions (which detectors contain
          which ancilla measurement record), not from ``LRBCircuit.back``.
          Audits every DEM mechanism of every circuit (30 sequences x 9
          depths, p = 2e-3) and compares with E's rule on every single
          mechanism, and on all mechanism pairs and 2e4 p = 0.02 samples of
          two circuits (sequence 0 at m = 6, sequence 1 at m = 22).

``c14``   tl_d5 vs unif=1 + final vs the pooled mld5 gap rule on fresh
          identical shots (p = 2e-3, m in {0, 6, 22}, 30 sequences).

``fair``  C6-C9: LRB-D data-2 policies against (i) the mld2 frontier on
          identical shots with exact per-shot ranking (no bins) and (ii) the
          frontier built only from the rounds the policy itself reads
          ("policy-information" frontier, own detectors between used rounds).

``exact`` Shot-noise-free pooled unif=1 + final acceptance and accepted error
          (lead's ``exact.exact_all_rounds``) at the c14 points, to
          adjudicate the sampled values.

Run on the shared allocation (heavy parts take a few minutes on 16 cores)::

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export PYTHONPATH=$PWD/src
    python studies/correction-decoders/qubit/scripts/verify_pareto.py all --workers 16
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

from collections import defaultdict  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import stim  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

DOCS = ROOT / "studies" / "correction-decoders" / "qubit"
GRID = DOCS / "results" / "grid_main"
E_RES = DOCS / "results" / "E"
SEED_BASE = 31_415_000_000      # disjoint from the grid (1..) and E (9e9)


def sequences() -> list[list[str]]:
    return json.loads((GRID / "meta.json").read_text())["sequences"]


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
            writer.writerow({k: (f"{v:.8g}" if isinstance(v, float) else v)
                             for k, v in row.items()})


def pool_map(func, tasks, workers):
    if workers <= 1:
        return [func(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(func, tasks))


# ---------------------------------------------------------------------------
# Own helpers (independent of acceptance_rules)
# ---------------------------------------------------------------------------

def dem_errors(dem: stim.DetectorErrorModel):
    """(probability, detector tuple, flips L0) for every error instruction."""
    out = []
    for ins in dem.flattened():
        if ins.type != "error":
            continue
        dets, flip = [], False
        for t in ins.targets_copy():
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id() and t.val == 0:
                flip = not flip
        out.append((ins.args_copy()[0], tuple(sorted(dets)), flip))
    return out


def brute_joint(dem: stim.DetectorErrorModel) -> np.ndarray:
    """Exact P(detectors, L) by Z2 convolution (own code, small DEMs)."""
    n = dem.num_detectors
    joint = np.zeros(2 ** (n + 1))
    joint[0] = 1.0
    index = np.arange(joint.size)
    for prob, dets, flip in dem_errors(dem):
        vec = 0
        for d in dets:
            vec ^= 1 << d
        if flip:
            vec ^= 1 << n
        if prob > 0 and vec:
            joint = (1.0 - prob) * joint + prob * joint[index ^ vec]
    return joint


def all_syndromes(n: int, start: int = 0, stop: int | None = None):
    stop = 2 ** n if stop is None else stop
    idx = np.arange(start, stop)
    return ((idx[:, None] >> np.arange(n)[None, :]) & 1).astype(bool)


def circuit_records(circuit: stim.Circuit):
    """Measured qubit per record, and per detector (coords, record set)."""
    meas_qubit: list[int] = []
    det_recs: list[frozenset] = []
    det_coords: list[tuple] = []
    for ins in circuit.flattened():
        if ins.name in ("M", "MZ", "MR", "MX", "MY"):
            for t in ins.targets_copy():
                meas_qubit.append(t.value)
        elif ins.name == "DETECTOR":
            k = len(meas_qubit)
            det_recs.append(frozenset(k + t.value for t in ins.targets_copy()))
            det_coords.append(tuple(ins.gate_args_copy()))
    return meas_qubit, det_recs, det_coords


class PooledFrontier:
    """Exact pooled gap rule: shots of all sequences ranked by |LLR|.

    The ranking (|LLR| rounded to 1e-9 so identical syndromes tie exactly)
    is computed once; sequence-bootstrap replicates reuse it with per-shot
    weights equal to the multiplicity of the shot's sequence, which is the
    same as re-pooling the resampled sequences.
    """

    def __init__(self, gap: np.ndarray, wrong: np.ndarray,
                 seq_of_shot: np.ndarray):
        g = np.round(np.asarray(gap, dtype=float), 9)
        self.order = np.argsort(-g, kind="stable")
        self.gs = g[self.order]
        self.wrong = np.asarray(wrong, dtype=float)[self.order]
        self.qexp = q_of(self.gs)
        self.seq = np.asarray(seq_of_shot)[self.order]
        # tie groups: [group_start[i], group_end[i]) for sorted position i
        change = np.flatnonzero(np.diff(self.gs) != 0) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [len(g)]])
        self.group_start = np.repeat(starts, ends - starts)
        self.group_end = np.repeat(ends, ends - starts)

    def _weights(self, mult):
        return (np.ones(len(self.gs)) if mult is None
                else np.asarray(mult, dtype=float)[self.seq])

    def error_at(self, n_accept: float, mult=None, expected=False) -> float:
        """Expected errors of the (randomised-boundary) gap rule accepting
        ``n_accept`` (weighted) shots."""
        w = self._weights(mult)
        e = (self.qexp if expected else self.wrong) * w
        cw = np.concatenate([[0.0], np.cumsum(w)])
        ce = np.concatenate([[0.0], np.cumsum(e)])
        if n_accept <= 0:
            return 0.0
        if n_accept >= cw[-1]:
            return float(ce[-1])
        i = int(np.searchsorted(cw, n_accept, side="left")) - 1
        i = min(max(i, 0), len(self.gs) - 1)
        s, t = self.group_start[i], self.group_end[i]
        frac = (n_accept - cw[s]) / (cw[t] - cw[s])
        return float(ce[s] + frac * (ce[t] - ce[s]))

    def acceptance_at(self, rate: float, mult=None, expected=False) -> float:
        """Largest (weighted) acceptance whose top-|LLR| set has error rate
        <= ``rate``, scanning whole tie groups (``expected`` uses q(|LLR|),
        the calibrated per-shot error, instead of realised errors)."""
        w = self._weights(mult)
        e = (self.qexp if expected else self.wrong) * w
        cw = np.cumsum(w)
        ce = np.cumsum(e)
        ends = np.flatnonzero(np.diff(np.concatenate([self.gs, [-1.0]]))
                              != 0)          # last index of each tie group
        feasible = (ce[ends] <= rate * cw[ends]) & (cw[ends] > 0)
        if not feasible.any():
            return 0.0
        j = int(np.flatnonzero(feasible)[-1])
        a0, e0 = cw[ends[j]], ce[ends[j]]
        if j + 1 < len(ends):                 # randomised next tie group
            da = cw[ends[j + 1]] - a0
            de = ce[ends[j + 1]] - e0
            if de - rate * da > 0 and da > 0:
                lam = min(1.0, (rate * a0 - e0) / (de - rate * da))
                return float(a0 + max(lam, 0.0) * da)
        return float(a0)


def q_of(gap) -> np.ndarray:
    with np.errstate(over="ignore"):
        return 1.0 / (1.0 + np.exp(np.abs(gap)))


def calibration_summary(gap: np.ndarray, wrong: np.ndarray, **label) -> dict:
    """Direct calibration test: in 0.5-wide |LLR| bins, observed MLD errors
    against sum_i q(|LLR_i|) (exact posterior => independent Bernoulli)."""
    edges = np.arange(0.0, 20.5, 0.5)
    k = len(edges) - 1
    bins = np.clip(np.searchsorted(edges, gap, side="right") - 1, 0, k - 1)
    q = q_of(gap)
    n = np.bincount(bins, minlength=k)
    obs = np.bincount(bins, weights=wrong.astype(float), minlength=k)
    exp = np.bincount(bins, weights=q, minlength=k)
    var = np.bincount(bins, weights=q * (1 - q), minlength=k)
    used = (n > 0) & (var > 0)
    z = (obs[used] - exp[used]) / np.sqrt(var[used])
    return {**label, "shots": int(n.sum()), "bins": int(used.sum()),
            "max_abs_z": float(np.max(np.abs(z))) if len(z) else np.nan,
            "bins_abs_z_gt3": int(np.sum(np.abs(z) > 3)),
            "observed_errors": float(obs.sum()),
            "expected_errors": float(exp.sum()),
            "total_z": float((obs.sum() - exp.sum()) / np.sqrt(var.sum()))}


# ---------------------------------------------------------------------------
# C4
# ---------------------------------------------------------------------------

def c4_sampled_task(task):
    p, m, si, seq, shots = task
    lc5 = Q.build_lrb_circuit(seq, m, p, terminal="data5")
    lc2 = Q.build_lrb_circuit(seq, m, p, terminal="data2")
    mld5 = M.TransferMatrixMLD(lc5.circuit.detector_error_model(),
                               M.lrb_detector_slots(lc5))
    mld2 = M.TransferMatrixMLD(lc2.circuit.detector_error_model(),
                               M.lrb_detector_slots(lc2))
    nr = 4 * lc5.num_rounds
    sampler = lc5.circuit.compile_detector_sampler(
        seed=SEED_BASE + 1000 * m + si)
    dets, obs = sampler.sample(shots, separate_observables=True)
    flip = obs[:, 0]
    raw = lc5.raw_syndromes(dets)
    unif1 = B.accept_mask(raw, lc5.rounds, B.used_rounds("unif", 1, m))
    unif1f = unif1 & ~dets[:, nr:].any(axis=1)
    rows = []
    for info, dec, cols, lrbd, name in (
            ("mld5", mld5, slice(None), unif1f, "unif=1+final"),
            ("mld2", mld2, slice(0, nr), unif1, "unif=1")):
        ev = dets[:, cols]
        llr = dec.log_likelihood_ratio(ev)
        g = np.abs(llr)
        zero = ~ev.any(axis=1)
        g0_values = np.unique(np.round(g[zero], 9))
        g0 = float(g0_values[0])
        other = g[~zero]
        gmax = float(other.max())
        margin = g0 - gmax
        n_syn = len(np.unique(np.packbits(ev, axis=1), axis=0))
        thr_set = g >= g0 - 0.5 * margin
        n_acc = int(lrbd.sum())
        order = np.argsort(-np.round(g, 9), kind="stable")
        topn = np.zeros(len(g), dtype=bool)
        topn[order[:n_acc]] = True
        strict = bool(np.round(g[order[n_acc - 1]], 9)
                      > np.round(g[order[n_acc]], 9))
        # E's identity test (paired difference and variance both zero) on
        # these shots, for comparison: z_i = e_rule - w_gap e_mld.
        wrong_mld = (llr < 0) != flip
        z = (lrbd & flip).astype(float) - topn * wrong_mld
        rows.append({
            "test": "sampled", "p": p, "depth": m, "seq": si, "info": info,
            "lrbd_policy": name, "shots": shots, "distinct_syndromes": n_syn,
            "zero_syndrome_llr_values": len(g0_values),
            "abs_llr_zero": g0, "max_abs_llr_other": gmax,
            "margin": margin, "lrbd_accepted": n_acc,
            "zero_syndrome_shots": int(zero.sum()),
            "threshold_set_equals_lrbd": bool(np.array_equal(thr_set, lrbd)),
            "top_n_equals_lrbd": bool(np.array_equal(topn, lrbd)),
            "boundary_strict": strict,
            "lrbd_errors": int((lrbd & flip).sum()),
            "mld_decision_on_zero_is_no_flip": bool((llr[zero] > 0).all()),
            "E_identity_test_passes": bool(abs(z.sum()) < 1e-9
                                           and np.sum(z * z) < 1e-9)})
    return rows


def c4_pairs_task(task):
    p, m, si, seq, terminal = task
    lc = Q.build_lrb_circuit(seq, m, p, terminal=terminal)
    dem = lc.circuit.detector_error_model()
    dec = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    n = dem.num_detectors
    vecs = []
    for _, dets, _ in dem_errors(dem):
        if dets:
            v = np.zeros(n, dtype=bool)
            v[list(dets)] = True
            vecs.append(v)
    vec = np.unique(np.array(vecs), axis=0)
    g0 = float(abs(dec.log_likelihood_ratio(np.zeros((1, n), bool))[0]))
    g1 = np.abs(dec.log_likelihood_ratio(vec))
    best, kind = float(g1.max()), "single"
    pairs = 0
    for i in range(len(vec) - 1):
        pv = vec[i] ^ vec[i + 1:]
        pv = pv[pv.any(axis=1)]
        if not len(pv):
            continue
        pairs += len(pv)
        g = float(np.abs(dec.log_likelihood_ratio(pv)).max())
        if g > best:
            best, kind = g, "pair"
    return {"test": "one_or_two_mechanisms", "p": p, "depth": m, "seq": si,
            "info": "mld5" if terminal == "data5" else "mld2",
            "detectors": n, "single_signatures": len(vec),
            "pair_syndromes": pairs, "abs_llr_zero": g0,
            "max_abs_llr_other": best, "argmax_kind": kind,
            "margin": g0 - best}


def c4_exhaustive_task(task):
    p, m, si, seq, terminal, own_bruteforce = task
    lc = Q.build_lrb_circuit(seq, m, p, terminal=terminal)
    dem = lc.circuit.detector_error_model()
    n = dem.num_detectors
    row = {"test": "exhaustive", "p": p, "depth": m, "seq": si,
           "info": "mld5" if terminal == "data5" else "mld2",
           "detectors": n, "syndromes": 2 ** n}
    if own_bruteforce:
        joint = brute_joint(dem)
        p0, p1 = joint[:2 ** n], joint[2 ** n:]
        llr_bf = np.log(p0) - np.log(p1)
        dec = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
        llr_tm = dec.log_likelihood_ratio(all_syndromes(n))
        g = np.abs(llr_bf)
        row["max_abs_diff_lead_vs_own_llr"] = float(np.max(np.abs(
            llr_tm - llr_bf)))
        row["abs_llr_zero"] = float(g[0])
        row["max_abs_llr_other"] = float(g[1:].max())
        j = 1 + int(np.argmax(g[1:]))
        row["argmax_syndrome_bits"] = format(j, f"0{n}b")[::-1]
        row["argmax_probability"] = float(p0[j] + p1[j])
    else:
        dec = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
        gmax, g0 = -1.0, None
        step = 1 << 18
        for start in range(0, 2 ** n, step):
            ev = all_syndromes(n, start, min(start + step, 2 ** n))
            g = np.abs(dec.log_likelihood_ratio(ev))
            if start == 0:
                g0 = float(g[0])
                g = g[1:]
            gmax = max(gmax, float(g[np.isfinite(g)].max()))
        row["abs_llr_zero"] = g0
        row["max_abs_llr_other"] = gmax
    row["margin"] = row["abs_llr_zero"] - row["max_abs_llr_other"]
    return row


def data2_is_marginal(seq, m=2, p=2e-3) -> dict:
    j5 = brute_joint(Q.build_lrb_circuit(seq, m, p, terminal="data5")
                     .circuit.detector_error_model())
    j2 = brute_joint(Q.build_lrb_circuit(seq, m, p, terminal="data2")
                     .circuit.detector_error_model())
    n5 = int(np.log2(len(j5))) - 1
    n2 = int(np.log2(len(j2))) - 1
    # index = detectors (bits 0..n-1) + L (bit n); final detectors are the
    # top two detector bits of data5.
    j5 = j5.reshape(2, 2 ** (n5 - n2), 2 ** n2)      # (L, final, rounds)
    marg = j5.sum(axis=1).reshape(-1)
    return {"test": "data2_dem_is_marginal_of_data5", "depth": m, "p": p,
            "max_abs_diff": float(np.max(np.abs(marg - j2))),
            "max_rel_diff": float(np.max(np.abs(marg - j2)
                                         / np.maximum(j2, 1e-300)))}


def cmd_c4(args, out: Path) -> None:
    seqs = sequences()
    rows = []
    t0 = time.time()
    tasks = [(2e-3, m, si, seqs[si], args.c4_shots)
             for m in (2, 6) for si in range(3)]
    for part in pool_map(c4_sampled_task, tasks, args.workers):
        rows += part
    print(f"[c4] sampled done {time.time() - t0:.0f}s", flush=True)
    ex = [(p, 2, si, seqs[si], term, True) for p in (2e-3,)
          for si in range(3) for term in ("data5", "data2")]
    ex += [(p, 4, si, seqs[si], term, False) for p in (2e-3, 1e-2)
           for si in range(3) for term in ("data5", "data2")]
    rows += pool_map(c4_exhaustive_task, ex, args.workers)
    print(f"[c4] exhaustive done {time.time() - t0:.0f}s", flush=True)
    pt = [(p, m, si, seqs[si], term) for p in (2e-3, 1e-2)
          for (m, si) in ((6, 0), (6, 1), (6, 2), (22, 0), (22, 1))
          for term in ("data5", "data2")]
    rows += pool_map(c4_pairs_task, pt, args.workers)
    rows.append(data2_is_marginal(seqs[0]))
    print(f"[c4] pairs done {time.time() - t0:.0f}s", flush=True)
    write_csv(out / "c4.csv", rows)
    for r in rows:
        print({k: r[k] for k in r if k in (
            "test", "p", "depth", "seq", "info", "abs_llr_zero",
            "max_abs_llr_other", "margin", "threshold_set_equals_lrbd",
            "top_n_equals_lrbd", "max_abs_diff", "max_abs_diff_lead_vs_own_llr")})


# ---------------------------------------------------------------------------
# C5 and C10
# ---------------------------------------------------------------------------

def undetectable_flip(dem) -> float:
    u = 0.0
    for prob, dets, flip in dem_errors(dem):
        if not dets and flip:
            u = u + prob - 2 * u * prob
    return u


def c5c10_task(task):
    p, m, si, seq = task
    rows = []
    for terminal in ("data5", "data2"):
        lc = Q.build_lrb_circuit(seq, m, p, terminal=terminal)
        dem = lc.circuit.detector_error_model()
        joint = brute_joint(dem)            # own exact P(s, L), m <= 4
        n = dem.num_detectors
        p0, p1 = joint[:2 ** n], joint[2 ** n:]
        u = undetectable_flip(dem)
        prob = p0 + p1
        q = np.minimum(p0, p1) / prob
        g = np.abs(np.log(p0) - np.log(p1))
        row = {"p": p, "depth": m, "seq": si, "terminal": terminal,
               "detectors": n, "u": u, "u_over_p": u / p,
               "max_abs_llr": float(g.max()),
               "argmax_is_zero_syndrome": bool(np.argmax(g) == 0)}
        if terminal == "data2":
            cap = np.log((1 - u) / u)
            row["cap_log_1mu_over_u"] = float(cap)
            row["cap_minus_max_abs_llr"] = float(cap - g.max())
            row["cap_holds"] = bool(g.max() <= cap + 1e-9)
            row["min_q"] = float(q.min())
            row["min_q_ge_u"] = bool(q.min() >= u * (1 - 1e-9))
        worst = -np.inf
        worst_f = np.inf
        for t in np.arange(0.0, 12.01, 0.25):
            acc = g >= t
            if not acc.any():
                continue
            eps = float((prob * q)[acc].sum() / prob[acc].sum())
            worst = max(worst, eps - 1.0 / (1.0 + np.exp(t)))
            worst_f = min(worst_f, abs(1 - 2 * eps) - np.tanh(t / 2))
        row["max_eps_minus_q_t"] = worst
        row["min_F_minus_tanh_half_t"] = worst_f
        row["F_ge_tanh_all_t"] = bool(worst_f >= -1e-12)
        rows.append(row)
    return rows


def cmd_c5c10(args, out: Path) -> None:
    seqs = sequences()
    tasks = [(p, m, si, seqs[si]) for p in (1e-4, 2e-3, 1e-2)
             for m in (0, 1, 2) for si in range(3)]
    tasks += [(p, 4, si, seqs[si]) for p in (2e-3,) for si in range(2)]
    rows = []
    for part in pool_map(c5c10_task, tasks, args.workers):
        rows += part
    # Grid re-check of the 793-cell claim from E's fits_threshold.csv.
    import pandas as pd
    fits = pd.read_csv(E_RES / "fits_threshold.csv")
    depth_cols = [c for c in fits.columns if c.startswith("F_m")]
    cells = viol = 0
    closest = np.inf
    for _, r in fits.iterrows():
        t = float(r["threshold"])
        if t <= 0:
            continue
        for c in depth_cols:
            f = r[c]
            sd = r["Fstd_" + c[2:]]
            if not np.isfinite(f):
                continue
            cells += 1
            bound = np.tanh(t / 2)
            if f < bound:
                viol += 1
            se = sd / np.sqrt(30) if sd > 0 else np.nan
            if np.isfinite(se) and se > 0:
                closest = min(closest, (f - bound) / se)
    rows.append({"p": "grid", "depth": "all", "seq": "all",
                 "terminal": "fits_threshold.csv", "grid_cells": cells,
                 "grid_cells_below_tanh": viol,
                 "closest_z_seq_se": closest})
    write_csv(out / "c5_c10.csv", rows)
    for r in rows:
        print(r)


# ---------------------------------------------------------------------------
# C12
# ---------------------------------------------------------------------------

STAB_STRINGS = ("ZZ_Z_", "_ZZ_Z", "XXX__", "_X_XX")


def _syndrome(ps: stim.PauliString) -> tuple:
    return tuple(int(not ps.commutes(stim.PauliString(s)))
                 for s in STAB_STRINGS)


def _in_stabilizer_group(ps: stim.PauliString) -> bool:
    gens = [stim.PauliString(s) for s in STAB_STRINGS]
    x, z = ps.to_numpy()
    for mask in range(16):
        prod = stim.PauliString(5)
        for i in range(4):
            if mask >> i & 1:
                prod *= gens[i]
        a, b = prod.to_numpy()
        if (a == x).all() and (b == z).all():
            return True
    return False


def cmd_c12(args, out: Path) -> None:
    rows = []
    xl, zl = stim.PauliString("X__X_"), stim.PauliString("Z_Z__")
    for one, two in (("_X___", "__XX_"), ("_Z___", "__ZZ_"),
                     ("_Y___", "__YY_")):
        a, b = stim.PauliString(one), stim.PauliString(two)
        prod = a * b
        rows.append({
            "test": "algebra", "qubit1": one, "fold_pair": two,
            "syndrome_qubit1": _syndrome(a), "syndrome_pair": _syndrome(b),
            "same_syndrome": _syndrome(a) == _syndrome(b),
            "product": str(prod),
            "product_commutes_with_stabilizers": _syndrome(prod)
            == (0, 0, 0, 0),
            "product_in_stabilizer_group": _in_stabilizer_group(prod),
            "product_anticommutes_XL": not prod.commutes(xl),
            "product_anticommutes_ZL": not prod.commutes(zl)})
    # X2X3 = X1 X_L S_X1 and Z2Z3 = Z1 Z_L S_Z1 (up to sign)
    for lhs, rhs in (("__XX_", stim.PauliString("_X___") * xl
                      * stim.PauliString("XXX__")),
                     ("__ZZ_", stim.PauliString("_Z___") * zl
                      * stim.PauliString("ZZ_Z_"))):
        a = stim.PauliString(lhs)
        x1, z1 = a.to_numpy()
        x2, z2 = rhs.to_numpy()
        rows.append({"test": "decomposition", "fold_pair": lhs,
                     "equals_up_to_sign": bool((x1 == x2).all()
                                               and (z1 == z2).all()),
                     "product": str(rhs)})
    # Circuit-level injection into noiseless circuits.
    seqs = sequences()
    total = matched = flips = 0
    q1_flips = 0
    per_gate = defaultdict(lambda: [0, 0])
    for si in range(5):
        for m in (2, 6, 10):
            flat = list(Q.build_lrb_circuit(seqs[si], m, 0.0,
                                            terminal="data5").circuit
                        .flattened())
            locs = [i for i, ins in enumerate(flat)
                    if ins.name in ("SWAP", "CZ") and sorted(
                        t.value for t in ins.targets_copy()) == [2, 3]]

            def run(at, targets):
                c = stim.Circuit()
                for i, ins in enumerate(flat):
                    c.append(ins)
                    if i == at:
                        c.append("CORRELATED_ERROR", targets, 1.0)
                d, o = c.compile_detector_sampler().sample(
                    1, separate_observables=True)
                return tuple(np.flatnonzero(d[0])), int(o[0, 0])

            for at in locs:
                gate = flat[at].name
                twins = {P: run(at, [getattr(stim, f"target_{P.lower()}")(1)])
                         for P in "XYZ"}
                q1_flips += sum(r[1] for r in twins.values())
                for P in "XYZ":
                    tgt = getattr(stim, f"target_{P.lower()}")
                    det, obs = run(at, [tgt(2), tgt(3)])
                    match = [k for k, r in twins.items() if r[0] == det]
                    total += 1
                    if len(match) == 1:
                        matched += 1
                        if twins[match[0]][1] != obs:
                            flips += 1
                            per_gate[gate][0] += 1
                    per_gate[gate][1] += 1
    rows.append({"test": "injection", "fault_locations": total // 3,
                 "pair_faults": total, "with_unique_qubit1_twin": matched,
                 "differ_in_observable": flips,
                 "qubit1_faults_flipping_observable": q1_flips,
                 "per_gate_flip_fraction": {g: v[0] / v[1]
                                            for g, v in per_gate.items()}})
    write_csv(out / "c12.csv", rows)
    for r in rows:
        print(r)


# ---------------------------------------------------------------------------
# C13: own tl_d5 rule
# ---------------------------------------------------------------------------

class OwnTimeLikeRule:
    """``tl_d5`` from its definition, built from detector definitions.

    Definition (E note, section 1.7): detection events are grouped into time
    slots (check rounds, then the final data-readout slot). A cluster is a
    maximal run of consecutive non-empty slots. A shot is accepted iff every
    cluster is exactly (i) the detection pattern of ONE flipped ancilla
    measurement, i.e. the set of detectors whose definition contains that
    measurement record, or (ii) a single terminal Z-check detector. No
    correction is applied.
    """

    def __init__(self, circuit: stim.Circuit, depth: int):
        meas_qubit, det_recs, det_coords = circuit_records(circuit)
        self.n = len(det_recs)
        self.slot = np.array([int(c[1]) for c in det_coords])
        self.bit = np.array([int(c[0]) for c in det_coords])
        self.num_slots = depth + 2
        assert set(self.slot.tolist()) == set(range(self.num_slots)), \
            "expected consecutive round labels 0..m+1"
        anc = [i for i, q in enumerate(meas_qubit) if q in (5, 6, 7, 8)]
        assert len(anc) == 4 * (depth + 1)
        sigs = set()
        for a in anc:
            sig = frozenset(d for d in range(self.n) if a in det_recs[d])
            assert sig, "an ancilla record that no detector uses"
            sigs.add(sig)
        lone_z = {frozenset([d]) for d in range(self.n)
                  if self.slot[d] == depth and self.bit[d] in (0, 1)}
        self.lone_z_subset_of_measurement_flips = lone_z <= sigs
        self.allowed = sigs | lone_z
        # Vectorized tables: 4-bit code per slot (bit = stabilizer index).
        self.single_ok = np.zeros((self.num_slots, 16), dtype=bool)
        self.pair_ok = np.zeros((self.num_slots, 256), dtype=bool)
        for sig in self.allowed:
            slots = sorted({int(self.slot[d]) for d in sig})
            codes = defaultdict(int)
            for d in sig:
                codes[int(self.slot[d])] ^= 1 << int(self.bit[d])
            if len(slots) == 1:
                self.single_ok[slots[0], codes[slots[0]]] = True
            else:
                assert len(slots) == 2 and slots[1] == slots[0] + 1
                t = slots[0]
                self.pair_ok[t, codes[t] | (codes[t + 1] << 4)] = True

    def accept_one(self, fired) -> bool:
        """Reference (set-based) implementation for one shot."""
        fired = sorted(set(int(d) for d in fired))
        if not fired:
            return True
        by_slot = defaultdict(set)
        for d in fired:
            by_slot[int(self.slot[d])].add(d)
        slots = sorted(by_slot)
        cluster: set = set()
        prev = None
        for t in slots:
            if prev is not None and t != prev + 1:
                if frozenset(cluster) not in self.allowed:
                    return False
                cluster = set()
            cluster |= by_slot[t]
            prev = t
        return frozenset(cluster) in self.allowed

    def codes(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=bool)
        out = np.zeros((dets.shape[0], self.num_slots), dtype=np.int64)
        for d in range(self.n):
            out[:, self.slot[d]] ^= dets[:, d].astype(np.int64) << self.bit[d]
        return out

    def accept(self, dets: np.ndarray) -> np.ndarray:
        c = self.codes(dets)
        occ = c != 0
        shots, T = c.shape
        ok = np.ones(shots, dtype=bool)
        false = np.zeros(shots, dtype=bool)
        for t in range(T):
            left = occ[:, t - 1] if t > 0 else false
            r1 = occ[:, t + 1] if t + 1 < T else false
            r2 = occ[:, t + 2] if t + 2 < T else false
            start = occ[:, t] & ~left
            ok &= ~(start & r1 & r2)
            lone = start & ~r1
            ok &= ~lone | self.single_ok[t][c[:, t]]
            if t + 1 < T:
                two = start & r1 & ~r2
                code = c[:, t] | (c[:, t + 1] << 4)
                ok &= ~two | self.pair_ok[t][code]
        return ok


def c13_task(task):
    p, m, si, seq, crosscheck = task
    lc5 = Q.build_lrb_circuit(seq, m, p, terminal="data5")
    dem = lc5.circuit.detector_error_model()
    rule = OwnTimeLikeRule(lc5.circuit, m)
    errs = dem_errors(dem)
    res = {"p": p, "depth": m, "seq": si, "mechanisms": len(errs),
           "rejected": 0.0, "accepted_ok": 0.0, "accepted_wrong": 0.0,
           "wrong_mechanisms": 0, "undetectable_flip": 0.0,
           "lone_z_subset_of_measurement_flips":
               rule.lone_z_subset_of_measurement_flips}
    vecs = np.zeros((len(errs), rule.n), dtype=bool)
    flips = np.zeros(len(errs), dtype=bool)
    accepted_ref = np.zeros(len(errs), dtype=bool)
    for i, (prob, dets, flip) in enumerate(errs):
        vecs[i, list(dets)] = True
        flips[i] = flip
        acc = rule.accept_one(dets)
        accepted_ref[i] = acc
        if not dets and flip:
            res["undetectable_flip"] += prob
        if not acc:
            res["rejected"] += prob
        elif flip:
            res["accepted_wrong"] += prob
            res["wrong_mechanisms"] += 1
        else:
            res["accepted_ok"] += prob
    res["vectorized_equals_reference_on_mechanisms"] = bool(
        np.array_equal(rule.accept(vecs), accepted_ref))
    if crosscheck:
        from lrb_qubit import acceptance_rules as A   # E's code, compare only
        layout = A.SlotLayout.from_circuit(lc5, True)
        tables = A.structural_tables(layout, False)
        e_acc, e_flip = A.evaluate_cluster_rule(layout.codes(vecs), layout,
                                                tables)
        res["E_rule_agrees_on_mechanisms"] = bool(
            np.array_equal(e_acc, accepted_ref))
        res["E_rule_flip_always_zero"] = bool(not e_flip.any())
        audit = A.single_fault_audit(layout, tables, dem)
        res["E_audit_rejected"] = audit["rejected"]
        res["E_audit_accepted_wrong"] = audit["accepted_wrong"]
        if crosscheck >= 2:
            # all pairs of distinct mechanism signatures
            uniq = np.unique(vecs[vecs.any(axis=1)], axis=0)
            dis = 0
            npairs = 0
            for i in range(len(uniq) - 1):
                pv = uniq[i] ^ uniq[i + 1:]
                mine = rule.accept(pv)
                theirs, _ = A.evaluate_cluster_rule(layout.codes(pv), layout,
                                                    tables)
                dis += int((mine != theirs).sum())
                npairs += len(pv)
            res["pair_syndromes_compared"] = npairs
            res["pair_disagreements"] = dis
            # high-p samples
            lc_hi = Q.build_lrb_circuit(seq, m, 0.02, terminal="data5")
            dets, _ = lc_hi.circuit.compile_detector_sampler(
                seed=SEED_BASE + 77 + 1000 * m + si).sample(
                    20000, separate_observables=True)
            mine = rule.accept(dets)
            theirs, _ = A.evaluate_cluster_rule(layout.codes(dets), layout,
                                                tables)
            ref = np.array([rule.accept_one(np.flatnonzero(r))
                            for r in dets[:3000]])
            res["sample_shots_compared"] = len(dets)
            res["sample_disagreements"] = int((mine != theirs).sum())
            res["sample_accept_rate_p0.02"] = float(mine.mean())
            res["vectorized_equals_reference_on_samples"] = bool(
                np.array_equal(ref, mine[:3000]))
    return res


def cmd_c13(args, out: Path) -> None:
    seqs = sequences()
    depths = [0, 2, 4, 6, 10, 14, 18, 20, 22]
    tasks = []
    for si in range(len(seqs)):
        for m in depths:
            level = 1
            if (si, m) in ((0, 6), (1, 22)):
                level = 2   # the two circuits audited in detail
            tasks.append((2e-3, m, si, seqs[si], level))
    rows = pool_map(c13_task, tasks, args.workers)
    write_csv(out / "c13_audit.csv", rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    print("detailed circuits:")
    print(df[df.get("pair_syndromes_compared").notna()].T.to_string())
    summ = df.groupby("depth")[["rejected", "accepted_wrong",
                                "E_audit_rejected"]].mean() / 2e-3
    print(summ.to_string())
    print("total accepted_wrong mechanisms:", int(df.wrong_mechanisms.sum()),
          " max accepted_wrong:", df.accepted_wrong.max(),
          " E agrees on all mechanisms:",
          bool(df.E_rule_agrees_on_mechanisms.all()),
          " vectorized==reference:",
          bool(df.vectorized_equals_reference_on_mechanisms.all()),
          " lone-Z subset of measurement flips:",
          bool(df.lone_z_subset_of_measurement_flips.all()))
    try:
        e = pd.read_csv(E_RES / "hybrid_audit.csv")
        e = e[(e.rule == "tl_d5") & np.isclose(e.p, 2e-3)]
        print("E hybrid_audit.csv tl_d5 p=2e-3 (rejected/p, accepted_wrong/p):")
        print(e[["depth", "rejected_over_p", "accepted_wrong_over_p"]]
              .to_string(index=False))
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# C14: tl_d5 vs unif=1 + final vs gap rule, identical fresh shots
# ---------------------------------------------------------------------------

def c14_task(task):
    p, m, si, seq, shots = task
    lc5 = Q.build_lrb_circuit(seq, m, p, terminal="data5")
    dem = lc5.circuit.detector_error_model()
    rule = OwnTimeLikeRule(lc5.circuit, m)
    mld5 = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc5))
    dets, obs = lc5.circuit.compile_detector_sampler(
        seed=SEED_BASE + 500_000 + 1000 * m + si).sample(
            shots, separate_observables=True)
    flip = obs[:, 0]
    tl = rule.accept(dets)
    u1f = ~dets.any(axis=1)
    llr = mld5.log_likelihood_ratio(dets)
    gap = np.abs(llr)
    wrong = (llr < 0) != flip
    # per-circuit: does the top-n gap rule at tl's acceptance pick tl's shots?
    n_tl = int(tl.sum())
    order = np.argsort(-np.round(gap, 9), kind="stable")
    topn = np.zeros(shots, dtype=bool)
    topn[order[:n_tl]] = True
    strict = (n_tl == 0 or n_tl == shots
              or np.round(gap[order[n_tl - 1]], 9)
              > np.round(gap[order[n_tl]], 9))
    # E's rule on the same shots (comparison only, not used for verdicts)
    from lrb_qubit import acceptance_rules as A
    layout = A.SlotLayout.from_circuit(lc5, True)
    e_acc, e_flip = A.evaluate_cluster_rule(
        layout.codes(dets), layout, A.structural_tables(layout, False))
    return {"seq": si, "shots": shots, "tl_acc": n_tl,
            "E_rule_disagreements": int((e_acc != tl).sum()),
            "E_rule_flips": int((e_acc & e_flip).sum()),
            "tl_err": int((tl & flip).sum()),
            "u1f_acc": int(u1f.sum()), "u1f_err": int((u1f & flip).sum()),
            "tl_set_is_top_n": bool(np.array_equal(topn, tl)) and bool(strict),
            "gap": gap.astype(np.float64), "wrong": wrong,
            "tl_mask_err_mld": int((tl & wrong).sum())}


def cmd_c14(args, out: Path) -> None:
    seqs = sequences()
    rng = np.random.default_rng(2718)
    rows, calib = [], []
    for m in args.c14_depths:
        t0 = time.time()
        tasks = [(2e-3, m, si, seqs[si], args.c14_shots)
                 for si in range(len(seqs))]
        res = pool_map(c14_task, tasks, args.workers)
        gap = np.concatenate([r["gap"] for r in res])
        wrong = np.concatenate([r["wrong"] for r in res])
        seq_of = np.concatenate([np.full(len(r["gap"]), i)
                                 for i, r in enumerate(res)])
        fr = PooledFrontier(gap, wrong, seq_of)
        n = len(gap)
        tl_a = np.array([r["tl_acc"] for r in res], float)
        tl_e = np.array([r["tl_err"] for r in res], float)
        u_a = np.array([r["u1f_acc"] for r in res], float)
        u_e = np.array([r["u1f_err"] for r in res], float)
        eps_tl = tl_e.sum() / tl_a.sum()
        gain_real = fr.acceptance_at(eps_tl) / tl_a.sum()
        gain_exp = fr.acceptance_at(eps_tl, expected=True) / tl_a.sum()
        eps_gap_same = fr.error_at(tl_a.sum()) / tl_a.sum()
        boot_gain, boot_ratio, boot_err = [], [], []
        for _ in range(args.n_boot):
            mult = np.bincount(rng.integers(0, len(res), len(res)),
                               minlength=len(res))
            ta, te, ua = mult @ tl_a, mult @ tl_e, mult @ u_a
            if ta == 0 or ua == 0:
                continue
            boot_ratio.append(ta / ua)
            boot_gain.append(fr.acceptance_at(te / ta, mult) / ta)
            ge = fr.error_at(ta, mult)
            boot_err.append((te / ta) / (ge / ta) if ge > 0 else np.nan)
        ci = lambda v: (float(np.nanquantile(v, 0.025)),  # noqa: E731
                        float(np.nanquantile(v, 0.975)))
        row = {"p": 2e-3, "depth": m, "shots": n,
               "tl_d5_acceptance": tl_a.sum() / n, "tl_d5_error": eps_tl,
               "unif1_final_acceptance": u_a.sum() / n,
               "unif1_final_error": u_e.sum() / u_a.sum(),
               "tl_over_unif1final_acceptance": tl_a.sum() / u_a.sum(),
               "tl_over_unif1final_ci": ci(boot_ratio),
               "gap_acc_gain_at_tl_error": gain_real,
               "gap_acc_gain_ci": ci(boot_gain),
               "gap_acc_gain_at_tl_error_expected_q": gain_exp,
               "gap_error_at_tl_acceptance": eps_gap_same,
               "tl_error_over_gap_error_same_acceptance":
                   eps_tl / eps_gap_same if eps_gap_same else np.nan,
               "tl_error_over_gap_error_ci": ci(boot_err),
               "circuits_tl_set_is_per_circuit_top_n":
                   sum(r["tl_set_is_top_n"] for r in res),
               "circuits": len(res),
               "mld_errors_on_tl_set": sum(r["tl_mask_err_mld"]
                                           for r in res),
               "E_rule_disagreements": sum(r["E_rule_disagreements"]
                                           for r in res),
               "E_rule_flips": sum(r["E_rule_flips"] for r in res)}
        rows.append(row)
        calib.append(calibration_summary(gap, wrong, p=2e-3, depth=m,
                                         decoder="mld5"))
        print(row, f"{time.time() - t0:.0f}s", flush=True)
        print(calib[-1], flush=True)
    write_csv(out / "c14.csv", rows)
    write_csv(out / "calibration_c14.csv", calib)


# ---------------------------------------------------------------------------
# Fairness of C6-C9: identical-shot frontiers with own information sets
# ---------------------------------------------------------------------------

def policy_info_circuit(lc5: "Q.LRBCircuit", used_labels: list[int]
                        ) -> stim.Circuit:
    """lc5's circuit with detectors replaced by difference detectors between
    consecutive USED rounds (raw detectors for the first used round). The
    observable and noise are unchanged; final data detectors are dropped.
    """
    meas_qubit, _, _ = circuit_records(lc5.circuit)
    anc = [i for i, q in enumerate(meas_qubit) if q in (5, 6, 7, 8)]
    # round position k owns ancilla records 4k..4k+3 in stabilizer order
    for k in range(lc5.num_rounds):
        assert [meas_qubit[r] for r in anc[4 * k:4 * k + 4]] == [5, 6, 7, 8]
    positions = [lc5.rounds.index(lbl) for lbl in used_labels]
    body = stim.Circuit()
    for ins in lc5.circuit.flattened():
        if ins.name != "DETECTOR":
            body.append(ins)
    total = len(meas_qubit)
    prev = None
    for slot, pos in enumerate(positions):
        if prev is None:
            subsets = tuple((s,) for s in range(4))
        else:
            ops = [op for seg in lc5.segments[prev + 1:pos + 1] for op in seg]
            subsets = Q._back_subsets(ops, range(4))
        for s in range(4):
            recs = [anc[4 * pos + s]]
            if prev is not None:
                recs += [anc[4 * prev + g] for g in subsets[s]]
            body.append("DETECTOR", [stim.target_rec(r - total)
                                     for r in recs], [s, slot])
        prev = pos
    return body


def fair_task(task):
    p, m, si, seq, shots, policies = task
    lc5 = Q.build_lrb_circuit(seq, m, p, terminal="data5")
    lc2 = Q.build_lrb_circuit(seq, m, p, terminal="data2")
    mld2 = M.TransferMatrixMLD(lc2.circuit.detector_error_model(),
                               M.lrb_detector_slots(lc2))
    meas = lc5.circuit.compile_sampler(
        seed=SEED_BASE + 900_000 + 1000 * m + si).sample(shots)
    dets, obs = lc5.circuit.compile_m2d_converter().convert(
        measurements=meas, separate_observables=True)
    flip = obs[:, 0]
    nr = 4 * lc5.num_rounds
    llr2 = mld2.log_likelihood_ratio(dets[:, :nr])
    raw = lc5.raw_syndromes(dets)
    out = {"seq": si, "gap2": np.abs(llr2), "wrong2": (llr2 < 0) != flip,
           "pol": {}}
    for policy, k in policies:
        used = B.used_rounds(policy, k, m)
        acc = B.accept_mask(raw, lc5.rounds, used)
        circ = policy_info_circuit(lc5, used)
        dem = circ.detector_error_model()
        pdets, pobs = circ.compile_m2d_converter().convert(
            measurements=meas, separate_observables=True)
        assert np.array_equal(pobs[:, 0], flip)
        coords = circ.get_detector_coordinates()
        slots = [int(coords[d][1]) for d in range(circ.num_detectors)]
        dec = M.TransferMatrixMLD(dem, slots)
        llr = dec.log_likelihood_ratio(pdets)
        zero = ~pdets.any(axis=1)
        g = np.abs(llr)
        g0 = np.round(g[zero], 9)
        zero_is_max = bool(len(g0) == 0 or (~zero).sum() == 0
                           or g0.min() > np.round(g[~zero].max(), 9))
        out["pol"][f"{policy}_{k}"] = {
            "acc": acc, "err": acc & flip,
            "acc_equals_zero_policy_syndrome": bool(np.array_equal(acc,
                                                                   zero)),
            "gapP": g, "wrongP": (llr < 0) != flip,
            "zero_is_max": zero_is_max}
    return out


def cmd_fair(args, out: Path) -> None:
    import pandas as pd
    seqs = sequences()
    policies = (("unif", 1), ("unif", 2), ("unif", 5), ("const", 1))
    e_mp = pd.read_csv(E_RES / "matched_points.csv")
    rng = np.random.default_rng(1618)
    rows, calib = [], []
    for p in args.fair_p:
        for m in args.fair_depths:
            t0 = time.time()
            tasks = [(p, m, si, seqs[si], args.fair_shots, policies)
                     for si in range(len(seqs))]
            res = pool_map(fair_task, tasks, args.workers)
            seq_of = np.concatenate([np.full(len(r["gap2"]), i)
                                     for i, r in enumerate(res)])
            gap2_all = np.concatenate([r["gap2"] for r in res])
            wrong2_all = np.concatenate([r["wrong2"] for r in res])
            fr2 = PooledFrontier(gap2_all, wrong2_all, seq_of)
            calib.append(calibration_summary(gap2_all, wrong2_all, p=p,
                                             depth=m, decoder="mld2"))
            for policy, k in policies[1:]:
                key = f"{policy}_{k}"
                calib.append(calibration_summary(
                    np.concatenate([r["pol"][key]["gapP"] for r in res]),
                    np.concatenate([r["pol"][key]["wrongP"] for r in res]),
                    p=p, depth=m, decoder=f"policy_info_{key}"))
            n = len(seq_of)
            mults = [np.bincount(rng.integers(0, len(res), len(res)),
                                 minlength=len(res))
                     for _ in range(args.n_boot)]
            for policy, k in policies:
                key = f"{policy}_{k}"
                acc_s = np.array([r["pol"][key]["acc"].sum() for r in res],
                                 float)
                err_s = np.array([r["pol"][key]["err"].sum() for r in res],
                                 float)
                frP = PooledFrontier(
                    np.concatenate([r["pol"][key]["gapP"] for r in res]),
                    np.concatenate([r["pol"][key]["wrongP"] for r in res]),
                    seq_of)
                a, e = acc_s.sum(), err_s.sum()
                if a == 0:
                    continue
                eps = e / a
                e2 = fr2.error_at(a) / a
                eP = frP.error_at(a) / a
                e2q = fr2.error_at(a, expected=True) / a
                ePq = frP.error_at(a, expected=True) / a
                gain2 = fr2.acceptance_at(eps) / a      # C8: equal error
                gainP = frP.acceptance_at(eps) / a
                boots2, bootsP = [], []
                for mult in mults:
                    ba, be = mult @ acc_s, mult @ err_s
                    if ba == 0:
                        continue
                    x2 = fr2.error_at(ba, mult)
                    xp = frP.error_at(ba, mult)
                    boots2.append(be / x2 if x2 > 0 else np.nan)
                    bootsP.append(be / xp if xp > 0 else np.nan)
                ci = lambda v: (  # noqa: E731
                    float(np.nanquantile(v, 0.025)),
                    float(np.nanquantile(v, 0.975)))
                e_row = e_mp[(e_mp.policy == f"lrbd_{key}")
                             & (e_mp.decoder == "mld2")
                             & (e_mp.depth == m) & np.isclose(e_mp.p, p)]
                rows.append({
                    "p": p, "depth": m, "policy": key, "shots": n,
                    "lrbd_acceptance": a / n, "lrbd_error": eps,
                    "acc_equals_zero_policy_syndrome_all_circuits": all(
                        r["pol"][key]["acc_equals_zero_policy_syndrome"]
                        for r in res),
                    "zero_syndrome_most_confident_policy_info_circuits":
                        sum(r["pol"][key]["zero_is_max"] for r in res),
                    "mld2_gap_error_same_acc": e2,
                    "ratio_vs_mld2_exact": eps / e2 if e2 else np.nan,
                    "ratio_vs_mld2_ci": ci(boots2),
                    "ratio_vs_mld2_expected_q": eps / e2q,
                    "policy_info_gap_error_same_acc": eP,
                    "ratio_vs_policy_info": eps / eP if eP else np.nan,
                    "ratio_vs_policy_info_ci": ci(bootsP),
                    "ratio_vs_policy_info_expected_q": eps / ePq,
                    "acc_gain_at_lrbd_error_mld2": gain2,
                    "acc_gain_at_lrbd_error_policy_info": gainP,
                    "E_grid_acc_gain_mld2":
                        float(e_row.acceptance_gain.iloc[0])
                        if len(e_row) else np.nan,
                    "E_grid_binned_ratio_vs_mld2":
                        float(e_row.error_reduction.iloc[0])
                        if len(e_row) else np.nan,
                    "E_grid_binned_ci": (
                        float(e_row.error_reduction_ci_low.iloc[0]),
                        float(e_row.error_reduction_ci_high.iloc[0]))
                    if len(e_row) else None,
                    "E_grid_lrbd_error": float(e_row.lrbd_error_rate.iloc[0])
                    if len(e_row) else np.nan})
                print({k2: rows[-1][k2] for k2 in (
                    "p", "depth", "policy", "lrbd_acceptance", "lrbd_error",
                    "ratio_vs_mld2_exact", "ratio_vs_mld2_ci",
                    "ratio_vs_policy_info", "E_grid_binned_ratio_vs_mld2",
                    "acc_gain_at_lrbd_error_mld2", "E_grid_acc_gain_mld2",
                    "acc_gain_at_lrbd_error_policy_info",
                    "zero_syndrome_most_confident_policy_info_circuits")},
                    flush=True)
            print(f"[fair] p={p} m={m} {time.time() - t0:.0f}s", flush=True)
    write_csv(out / "fair.csv", rows)
    write_csv(out / "calibration_fair.csv", calib)
    for c in calib:
        print(c)


def cmd_exact(args, out: Path) -> None:
    from lrb_qubit import exact as X
    seqs = sequences()
    rows = []
    for m in args.c14_depths:
        acc = err = 0.0
        for seq in seqs:
            lc = Q.build_lrb_circuit(seq, m, 2e-3, terminal="data5")
            dec = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                      M.lrb_detector_slots(lc))
            a, f = X.exact_all_rounds(dec, lc.num_rounds, True)
            acc += a
            err += f
        rows.append({"p": 2e-3, "depth": m, "policy": "unif_1_final",
                     "exact_pooled_acceptance": acc / len(seqs),
                     "exact_pooled_error": err / acc})
        print(rows[-1], flush=True)
    write_csv(out / "exact_unif1_final.csv", rows)


# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cmd", choices=("c4", "c5c10", "c12", "c13", "c14",
                                        "fair", "exact", "all"))
    parser.add_argument("--out", default=str(DOCS / "results" / "E-verify"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--c4-shots", type=int, default=200_000)
    parser.add_argument("--c14-shots", type=int, default=100_000)
    parser.add_argument("--c14-depths", default="0,6,22")
    parser.add_argument("--fair-shots", type=int, default=100_000)
    parser.add_argument("--fair-p", default="0.001,0.002")
    parser.add_argument("--fair-depths", default="6,22")
    parser.add_argument("--n-boot", type=int, default=200)
    args = parser.parse_args()
    args.c14_depths = [int(x) for x in args.c14_depths.split(",")]
    args.fair_p = [float(x) for x in args.fair_p.split(",")]
    args.fair_depths = [int(x) for x in args.fair_depths.split(",")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cmds = {"c4": cmd_c4, "c5c10": cmd_c5c10, "c12": cmd_c12,
            "c13": cmd_c13, "c14": cmd_c14, "fair": cmd_fair,
            "exact": cmd_exact}
    todo = list(cmds) if args.cmd == "all" else [args.cmd]
    for name in todo:
        t0 = time.time()
        print(f"=== {name}", flush=True)
        cmds[name](args, out)
        print(f"=== {name} done in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
