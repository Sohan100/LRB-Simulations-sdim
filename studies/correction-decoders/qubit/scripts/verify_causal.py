"""Adversarial verification of work package D (causal vs offline decoding, Pauli frame).

Independent of ``src/lrb_qubit/decoders_causal.py`` except where the
implementer's decoder is the object under test (tamper test, shot-by-shot
agreement). The lead's circuit builder (``circuits.build_lrb_circuit``) and exact
MLD (``mld.TransferMatrixMLD``) are the reference model, as instructed.

Subcommands
-----------
``tamper``
    Causality of the implementer's committed decoders (``CausalWindowMLD``,
    state and logical commit, w = 0, 1, 2, 4). For every slot t: randomise,
    flip or set the detection events of slots > t + w and check that no commit
    for slots <= t changes; power check (changing slot t + w alone does change
    the commit of slot t); negative control (a w + 1 decoder must fail the
    w test); streaming timing (slot t is committed exactly when slot t + w is
    pushed). The same harness is run on my own decoder.
``c1``
    My own committed (state) causal decoder. For slot t it builds, by brute-force
    Z2 convolution, the exact joint distribution of the window
    (slot t-1 leftover, slots t..t+w, b_t, Delta l_t) over the DEM mechanisms
    whose latest slot lies in [t, t+w+1]; mechanisms with latest slot <= t-1
    are summarised by the committed state (b_{t-1}, l_{t-1}), whose only
    effect is to replace the observed slot t-1 bits by s_{t-1} XOR b_{t-1}.
    It commits the posterior mode of (l_t, b_t); exact ties in l_t are broken
    by an explicit rule (``l0``, ``keep``, ``random``, ``l1``, ``flip``; see
    ``WindowCommitDecoder.decode``), because the tie rule changes the result.
    No message passing, no code shared with decoders_causal.py. Default: five
    NEW sequences, p = 5e-3, all depths 0..22, w = 0 and 1; ``--study`` uses
    the implementer's 30 sequences instead (independent shots). The same
    shots are decoded by the lead's offline MLD and by the implementer's
    decoders (w = 0, 1, 4, inf) for shot-by-shot agreement.
``c10``
    Pauli frame versus physical feedback. (a) Exact: four linear rules as Stim
    classically controlled Paulis (``CX/CY/CZ rec[-k] q``) versus the plain
    circuit whose observable is the frame-corrected readout; the frame
    coefficients come from my own Pauli propagation through the noiseless
    circuit (stim.PauliString.after, op by op). (b) Rule A on three study
    sequences, compared with the implementer's ``frame_rule_a_exact.csv``.
    (c) Non-linear AND rules ((1,1) X syndrome -> Z1, (1,1) Z syndrome -> X1;
    memoryless and two-round) with ``stim.FlipSimulator``: physical injection
    versus frame, same seed, checked shot by shot on the output AND on every
    measurement record.
``c3``
    Per-slot filter versus smoother of the logical frame l_t at depth 22, both
    computed with the lead's exact MLD on modified DEMs (filter: DEM truncated
    to slots <= t + w with mechanisms of latest slot t + w + 1 cut at the
    window edge; smoother: full DEM whose observable counts only mechanisms
    with latest slot <= t). Ground truth from DEM error sampling.
``report``
    Fits, ratios and summaries into ``results/D-verify/``.

Run (shared allocation, at most 16 workers):

    srun --jobid=<JOBID> --overlap --cpu-bind=none -N 1 -n 1 -c 256 bash -c \
"module load python/3.11; cd <WORKTREE>; export OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=<WORKTREE>/src; \
python studies/correction-decoders/qubit/scripts/verify_causal.py c1 --workers 16"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_v) != "1" for _v in _THREAD_VARS):
    for _v in _THREAD_VARS:
        os.environ[_v] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import stim  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

OUT = ROOT / "studies/correction-decoders/qubit/results/D-verify"
STUDY_SEED = 20261028
NEW_SEED = 990_000
NUM_QUBITS = 9


# ---------------------------------------------------------------------------
# Common helpers
# ---------------------------------------------------------------------------

def new_sequences(n: int, base: int = NEW_SEED, length: int = 22):
    """Fresh sequences, guaranteed different from the 30 study sequences."""
    study = E.clifford_sequences(30, 22, STUDY_SEED)
    seqs = []
    for i in range(n):
        rng = np.random.default_rng(base + i)
        seq = [C.SINGLE_QUBIT_CLIFFORDS[int(k)]
               for k in rng.integers(0, len(C.SINGLE_QUBIT_CLIFFORDS),
                                     size=length)]
        assert all(seq != s for s in study)
        seqs.append(seq)
    return seqs


def slots_from_coords(circuit: stim.Circuit) -> np.ndarray:
    """Slot of each detector = its round label (detector coordinate 1).

    For the all-rounds data5 circuit the labels are 0..m-1 (forward), m
    (terminal) and m+1 (final data detectors), i.e. already slot indices.
    """
    coords = circuit.get_detector_coordinates()
    slots = np.array([int(round(coords[d][1]))
                      for d in range(circuit.num_detectors)], dtype=np.int64)
    assert np.all(np.diff(slots) >= 0)
    assert set(slots.tolist()) == set(range(int(slots.max()) + 1))
    return slots


def parse_dem(dem: stim.DetectorErrorModel):
    """[(probability, detectors, flips_L0)] in DEM instruction order."""
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        dets, obs = [], 0
        for tgt in inst.targets_copy():
            if tgt.is_relative_detector_id():
                dets.append(tgt.val)
            elif tgt.is_logical_observable_id() and tgt.val == 0:
                obs ^= 1
        out.append((prob, tuple(dets), obs))
    return out


def latest_slots(mechs, slots):
    u = np.full(len(mechs), -1, dtype=np.int64)
    for k, (_, dets, obs) in enumerate(mechs):
        if dets:
            ks = slots[list(dets)]
            if ks.min() < ks.max() - 1:
                raise ValueError("mechanism spans non-adjacent slots")
            u[k] = int(ks.max())
        elif obs:
            raise ValueError("undetectable logical mechanism")
    return u


def write_rows(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
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
# My committed causal decoder (brute-force window joints)
# ---------------------------------------------------------------------------

class WindowCommitDecoder:
    """Committed (state) causal decoder with lookahead ``w``, brute force.

    At slot t the decoder holds a committed (b_{t-1}, l_{t-1}). The unknown
    mechanisms that can influence slots t-1..t+w are those whose latest slot
    lies in [t, t+w+1]; mechanisms with latest slot <= t-1 are summarised by
    the committed state. The table built here is the exact joint
    P(X_{t-1}, X_t, ..., X_{t+w}, B_t, DL_t) of those mechanisms, where X_k is
    their contribution to slot k (slot t+w+1 is marginalised), B_t the slot-t
    bits of mechanisms with latest slot t and DL_t their observable flips.
    Conditioning: X_{t-1} = s_{t-1} XOR b_{t-1}, X_k = s_k. The decision is the
    mode over (l_t = l_{t-1} XOR DL_t, b_t), ties broken towards the smallest
    index l_t * 2**n_t + b_t (values within relative ``tol`` count as ties).
    """

    def __init__(self, dem, slots, w: int, tol: float = 1e-12):
        slots = np.asarray(slots, dtype=np.int64)
        self.w = int(w)
        self.tol = tol
        self.T = int(slots.max()) + 1
        self.slot_dets = [np.flatnonzero(slots == k) for k in range(self.T)]
        self.sizes = [len(d) for d in self.slot_dets]
        pos = np.zeros(len(slots), dtype=np.int64)
        for k in range(self.T):
            pos[self.slot_dets[k]] = np.arange(self.sizes[k])
        mechs = parse_dem(dem)
        u = latest_slots(mechs, slots)
        self.mechs = [(prob, [(int(slots[d]), int(pos[d])) for d in dets],
                       obs, int(uk))
                      for (prob, dets, obs), uk in zip(mechs, u) if uk >= 0]
        self.tables = [self._table(t) for t in range(self.T)]

    def _table(self, t: int):
        end = min(t + self.w, self.T - 1)
        window = ([t - 1] if t > 0 else []) + list(range(t, end + 1))
        off, nb = {}, 0
        for k in window:
            off[k] = nb
            nb += self.sizes[k]
        n_obs = nb
        b_off = nb
        nb += self.sizes[t]
        l_bit = nb
        nb += 1
        vec_prob: dict[int, float] = {}
        for prob, dets, obs, uk in self.mechs:
            if uk < t or uk > end + 1:
                continue
            v = 0
            for k, i in dets:
                if k in off:
                    v ^= 1 << (off[k] + i)
                if uk == t and k == t:
                    v ^= 1 << (b_off + i)
            if obs and uk == t:
                v ^= 1 << l_bit
            if v == 0:
                continue
            q = vec_prob.get(v, 0.0)
            vec_prob[v] = q + prob - 2.0 * q * prob
        joint = np.zeros(2 ** nb)
        joint[0] = 1.0
        index = np.arange(2 ** nb)
        for v, prob in vec_prob.items():
            joint = (1.0 - prob) * joint + prob * joint[index ^ v]
        table = joint.reshape(2, 2 ** self.sizes[t], 2 ** n_obs)  # [dl, b, x]
        return {"table": table, "off": off}

    def slot_values(self, dets: np.ndarray) -> np.ndarray:
        dets = np.asarray(dets, dtype=bool)
        out = np.zeros((dets.shape[0], self.T), dtype=np.int64)
        for k, d in enumerate(self.slot_dets):
            out[:, k] = dets[:, d].astype(np.int64) @ (1 << np.arange(len(d)))
        return out

    TIE_RULES = ("l0", "keep", "random", "l1", "flip")

    def decode(self, dets: np.ndarray, return_history: bool = False,
               tie: str = "l0", seed: int = 0):
        """Commit slot by slot.

        ``tie`` decides between exactly tied logical frames (posterior values
        within relative ``tol``); ties in b_t always go to the smallest b_t:
        ``l0`` prefers l_t = 0 (= numpy's first index, the implementer's
        rule when its floats tie exactly), ``l1`` prefers l_t = 1, ``keep``
        prefers l_t = l_{t-1}, ``flip`` prefers l_t != l_{t-1}, ``random``
        picks uniformly.
        """
        if tie not in self.TIE_RULES:
            raise ValueError(tie)
        rng = np.random.default_rng(seed)
        vals = self.slot_values(dets)
        shots = vals.shape[0]
        idx = np.arange(shots)
        l_hat = np.zeros(shots, dtype=np.int64)
        b_hat = np.zeros(shots, dtype=np.int64)
        l_hist = np.zeros((shots, self.T), dtype=np.int64)
        b_hist = np.zeros((shots, self.T), dtype=np.int64)
        gap = np.zeros((shots, self.T))
        self.dead = 0
        self.ties = 0
        self.l_ties = 0
        for t in range(self.T):
            tab = self.tables[t]
            x = np.zeros(shots, dtype=np.int64)
            for k, o in tab["off"].items():
                col = vals[:, k] ^ b_hat if k == t - 1 else vals[:, k]
                x |= col << o
            post = np.moveaxis(tab["table"][:, :, x], 2, 0).copy()  # (s,dl,b)
            flip = l_hat == 1
            post[flip] = post[flip][:, ::-1, :]                     # -> (s,l,b)
            n_b = 2 ** self.sizes[t]
            flat = post.reshape(shots, -1)
            top = flat.max(axis=1)
            self.dead += int((top <= 0).sum())
            cand = flat >= top[:, None] * (1.0 - self.tol)
            self.ties += int((cand.sum(axis=1) > 1).sum())
            l_tied = cand[:, :n_b].any(axis=1) & cand[:, n_b:].any(axis=1)
            self.l_ties += int(l_tied.sum())
            if tie == "l0":
                pref = np.zeros(shots, dtype=np.int64)
            elif tie == "l1":
                pref = np.ones(shots, dtype=np.int64)
            elif tie == "keep":
                pref = l_hat.copy()
            elif tie == "flip":
                pref = 1 - l_hat
            else:
                pref = rng.integers(0, 2, size=shots)
            c3 = cand.reshape(shots, 2, n_b)
            first = c3[idx, pref]                 # preferred-l block first
            second = c3[idx, 1 - pref]
            in_first = first.any(axis=1)
            l_new = np.where(in_first, pref, 1 - pref)
            b_new = np.where(in_first, first.argmax(axis=1),
                             second.argmax(axis=1))
            srt = np.sort(flat, axis=1)
            gap[:, t] = 1.0 - srt[:, -2] / np.where(top > 0, top, 1.0)
            l_hat = l_new.astype(np.int64)
            b_hat = b_new.astype(np.int64)
            l_hist[:, t] = l_hat
            b_hist[:, t] = b_hat
        pred = l_hat.astype(bool)
        if return_history:
            return pred, l_hist, b_hist, gap
        return pred


# ---------------------------------------------------------------------------
# tamper: causality of the implementer's committed decoders
# ---------------------------------------------------------------------------

def _impl_histories(dec, dets):
    pred, l_hist, b_hist = dec.decode(dets, return_history=True)
    return pred, l_hist.astype(np.int64), b_hist.astype(np.int64)


def _mine_histories(dec, dets):
    pred, l_hist, b_hist, _ = dec.decode(dets, return_history=True)
    return pred, l_hist, b_hist


def cmd_tamper(args) -> int:
    from lrb_qubit import decoders_causal as D
    rng = np.random.default_rng(31337)
    seq = new_sequences(1)[0]
    rows = []
    for depth in (6, 12):
        lc = Q.build_lrb_circuit(seq, depth, args.p, terminal="data5")
        dem = lc.circuit.detector_error_model()
        slots = slots_from_coords(lc.circuit)
        last = int(slots.max())
        dets, _ = lc.circuit.compile_detector_sampler(seed=depth).sample(
            args.shots, separate_observables=True)
        variants = []
        for w in (0, 1, 2, 4):
            for commit in ("state", "logical"):
                variants.append((f"impl_{commit}_w{w}", w,
                                 D.CausalWindowMLD(dem, slots, w, commit),
                                 D.CausalWindowMLD(dem, slots, w + 1, commit),
                                 _impl_histories))
            if w <= 1:
                variants.append((f"mine_state_w{w}", w,
                                 WindowCommitDecoder(dem, slots, w),
                                 WindowCommitDecoder(dem, slots, w + 1),
                                 _mine_histories))
        for name, w, dec, leaky, hist in variants:
            _, l_ref, b_ref = hist(dec, dets)
            _, l_lk, b_lk = hist(leaky, dets)
            violations = 0
            leaky_violations = 0
            power = []
            boundary_changes = 0
            checks = 0
            for t in range(last + 1):
                horizon = t + w
                later = slots > horizon
                if later.any():
                    for mode in ("random", "flip", "ones"):
                        tam = dets.copy()
                        if mode == "random":
                            tam[:, later] = rng.random(
                                (len(dets), int(later.sum()))) < 0.5
                        elif mode == "flip":
                            tam[:, later] ^= True
                        else:
                            tam[:, later] = True
                        _, l2, b2 = hist(dec, tam)
                        bad = ((l2[:, :t + 1] != l_ref[:, :t + 1]).any(axis=1)
                               | (b2[:, :t + 1] != b_ref[:, :t + 1]).any(axis=1))
                        violations += int(bad.sum())
                        _, l3, b3 = hist(leaky, tam)
                        bad = ((l3[:, :t + 1] != l_lk[:, :t + 1]).any(axis=1)
                               | (b3[:, :t + 1] != b_lk[:, :t + 1]).any(axis=1))
                        leaky_violations += int(bad.sum())
                        checks += 1
                    # Only the first slot beyond the window changes.
                    edge = slots == horizon + 1
                    tam = dets.copy()
                    tam[:, edge] ^= rng.random((len(dets), int(edge.sum()))) < 0.5
                    _, l2, b2 = hist(dec, tam)
                    boundary_changes += int(
                        ((l2[:, :t + 1] != l_ref[:, :t + 1]).any(axis=1)
                         | (b2[:, :t + 1] != b_ref[:, :t + 1]).any(axis=1)).sum())
                # Power: the last slot inside the window does matter.
                inside = slots == min(horizon, last)
                tam = dets.copy()
                tam[:, inside] ^= rng.random((len(dets), int(inside.sum()))) < 0.5
                _, l2, b2 = hist(dec, tam)
                power.append(float(((l2[:, t] != l_ref[:, t])
                                    | (b2[:, t] != b_ref[:, t])).mean()))
            rows.append({"depth": depth, "p": args.p, "decoder": name,
                         "lookahead": w, "shots": args.shots,
                         "tamper_patterns": checks,
                         "violations_slots_beyond_t_plus_w": violations,
                         "violations_edge_slot_t_plus_w_plus_1":
                             boundary_changes,
                         "negative_control_w_plus_1_violations":
                             leaky_violations,
                         "power_min_frac_commit_changes_when_slot_t_plus_w_changes":
                             min(power),
                         "power_mean": float(np.mean(power))})
            print(rows[-1], flush=True)
        # Streaming timing of the implementer's stream.
        for w in (0, 1, 2, 4):
            dec = D.CausalWindowMLD(dem, slots, w, "state")
            values = dec.slot_values(dets)
            stream = dec.stream(len(dets), record=True)
            timing_ok = True
            snap = {}
            for k in range(last + 1):
                newly = stream.push(values[:, k])
                # Slot t may be committed only once slot min(t + w, last)
                # has arrived, and must be committed right then.
                expect = [t for t in range(last + 1)
                          if min(t + w, last) == k]
                timing_ok &= newly == expect
                for t in newly:
                    snap[t] = (stream.l_hist[:, t].copy(),
                               stream.b_hist[:, t].copy())
            final_same = all(
                np.array_equal(lv, stream.l_hist[:, t])
                and np.array_equal(bv, stream.b_hist[:, t])
                for t, (lv, bv) in snap.items())
            rows.append({"depth": depth, "p": args.p,
                         "decoder": f"impl_state_w{w}_stream",
                         "lookahead": w, "stream_commit_timing_ok": timing_ok,
                         "commits_unchanged_after_commit": final_same,
                         "stream_equals_batch": bool(np.array_equal(
                             stream.prediction, dec.decode(dets)))})
            print(rows[-1], flush=True)
    write_rows(Path(args.out) / "tamper.csv", rows)
    return 0


# ---------------------------------------------------------------------------
# c1: independent reproduction of the committed-decoder ratios
# ---------------------------------------------------------------------------

def _mine_name(w, tie):
    return f"mine_w{w}" if tie == "l0" else f"mine_w{w}_{tie}"


C1_DECODERS = (["offline"]
               + [_mine_name(w, tie) for w in (0, 1)
                  for tie in WindowCommitDecoder.TIE_RULES]
               + ["impl_w0", "impl_w1", "impl_w4", "impl_winf"])


def _c1_task(args):
    seq_index, seq, depth, p, shots, seed = args
    from lrb_qubit import decoders_causal as D
    t0 = time.perf_counter()
    lc = Q.build_lrb_circuit(seq, depth, p, terminal="data5")
    slots = slots_from_coords(lc.circuit)
    assert np.array_equal(slots, M.lrb_detector_slots(lc))
    dem = lc.circuit.detector_error_model()
    dets, obs = lc.circuit.compile_detector_sampler(seed=seed).sample(
        shots, separate_observables=True)
    flip = obs[:, 0]
    base = M.TransferMatrixMLD(dem, slots)
    preds, hist = {}, {}
    preds["offline"] = base.log_likelihood_ratio(dets) < 0
    info = {}
    for w in (0, 1):
        mine = WindowCommitDecoder(dem, slots, w)
        for tie in mine.TIE_RULES:
            name = _mine_name(w, tie)
            if tie == "l0":
                pred, lh, bh, gap = mine.decode(dets, return_history=True)
                hist[name] = (lh, bh, gap)
                info[f"{name}_l_tie_commits"] = mine.l_ties
                info[f"{name}_any_tie_commits"] = mine.ties
            else:
                pred = mine.decode(dets, tie=tie, seed=seed + 31 * w)
            preds[name] = pred
            info[f"{name}_dead"] = mine.dead
    for name, w in (("impl_w0", 0), ("impl_w1", 1), ("impl_w4", 4),
                    ("impl_winf", None)):
        dec = D.CausalWindowMLD(None, None, w, "state", base=base)
        if w in (0, 1):
            pred, lh, bh = dec.decode(dets, return_history=True)
            hist[name] = (lh.astype(np.int64), bh.astype(np.int64))
        else:
            pred = dec.decode(dets)
        preds[name] = pred
        info[f"{name}_dead"] = dec.dead_shots
    errors = {k: int((v != flip).sum()) for k, v in preds.items()}
    wrong_off = preds["offline"] != flip
    n10 = {k: int(((v != flip) & ~wrong_off).sum()) for k, v in preds.items()}
    n01 = {k: int(((v == flip) & wrong_off).sum()) for k, v in preds.items()}
    agree = {}
    for w in (0, 1):
        a, b = preds[f"mine_w{w}"], preds[f"impl_w{w}"]
        agree[f"pred_disagree_w{w}"] = int((a != b).sum())
        lm, bm, gap = hist[f"mine_w{w}"]
        li, bi = hist[f"impl_w{w}"]
        diff = (lm != li) | (bm != bi)
        rows_diff = np.flatnonzero(diff.any(axis=1))
        agree[f"history_disagree_shots_w{w}"] = int(len(rows_diff))
        if len(rows_diff):
            first = diff[rows_diff].argmax(axis=1)
            agree[f"max_rel_gap_at_first_divergence_w{w}"] = float(
                gap[rows_diff, first].max())
        else:
            agree[f"max_rel_gap_at_first_divergence_w{w}"] = 0.0
    agree["pred_disagree_w4_vs_winf"] = int(
        (preds["impl_w4"] != preds["impl_winf"]).sum())
    return {"seq": seq_index, "depth": depth, "shots": shots,
            "errors": errors, "n10": n10, "n01": n01, "agree": agree,
            "info": info, "seconds": time.perf_counter() - t0}


def cmd_c1(args) -> int:
    if args.study:
        seqs = [list(q) for q in
                E.clifford_sequences(30, 22, STUDY_SEED)[:args.num_seq]]
    else:
        seqs = new_sequences(args.num_seq)
    tag = "c1_study" if args.study else "c1"
    depths = list(range(23)) if args.depths == "all" else [
        int(d) for d in args.depths.split(",")]
    tasks = [(i, seqs[i], d, args.p, args.shots,
              555_000 + 1000 * i + d + int(round(args.p * 1e7)))
             for i in range(args.num_seq) for d in depths]
    tasks.sort(key=lambda t: -t[2])          # long tasks first
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results = []
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(_c1_task, t) for t in tasks]
        for n, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            results.append(res)
            print(f"[{time.perf_counter() - start:6.1f}s] seq {res['seq']} "
                  f"m={res['depth']} ({res['seconds']:.1f}s) {n}/{len(tasks)} "
                  f"{res['agree']}", flush=True)
    meta = {"p": args.p, "depths": depths, "num_seq": args.num_seq,
            "shots": args.shots,
            "sequence_source": (f"study seed {STUDY_SEED}" if args.study
                                else f"new seed {NEW_SEED}"),
            "sequences": seqs, "decoders": C1_DECODERS}
    path = out / f"{tag}_p{args.p:g}.json"
    with open(path, "w") as handle:
        json.dump({"meta": meta, "results": results}, handle)
    print(f"wrote {path}", flush=True)
    return 0


# ---------------------------------------------------------------------------
# c10: Pauli frame versus physical feedback
# ---------------------------------------------------------------------------

_SKIP = {"DETECTOR", "OBSERVABLE_INCLUDE", "TICK", "SHIFT_COORDS",
         "QUBIT_COORDS"}


def round_structure(circuit: stim.Circuit):
    """Instructions without annotations, plus per check round the index of
    the instruction measuring ancilla 8 (end of the round) and the record
    indices of M5, M6, M7, M8; also the final data-readout records."""
    insts = [inst for inst in circuit.flattened() if inst.name not in _SKIP]
    rounds, cur, m = [], {}, 0
    data_rec = {}
    for i, inst in enumerate(insts):
        if inst.name == "M":
            targets = [t.value for t in inst.targets_copy()]
            for j, q in enumerate(targets):
                if q in (5, 6, 7, 8):
                    cur[q] = m
                    if q == 8:
                        assert j == len(targets) - 1
                        rounds.append((i, [cur[5], cur[6], cur[7], cur[8]]))
                        cur = {}
                else:
                    data_rec[q] = m
                m += 1
        elif inst.name in ("MR", "MX", "MY", "MZ", "MPP"):
            raise ValueError(f"unexpected measurement {inst.name}")
    return insts, rounds, m, data_rec


def pauli_flips(insts0, start: int, m_start: int, pauli: str, qubit: int):
    """Measurement records flipped by ``pauli`` on ``qubit`` applied right
    after instruction ``start`` of the NOISELESS circuit (frame semantics:
    a Z-basis measurement flips if the propagated Pauli has X or Y on it;
    a reset removes the component)."""
    P = stim.PauliString(NUM_QUBITS)
    P[qubit] = pauli
    m = m_start
    flipped = set()
    for inst in insts0[start + 1:]:
        if inst.name == "M":
            for tgt in inst.targets_copy():
                if P[tgt.value] in (1, 2):
                    flipped.add(m)
                m += 1
        elif inst.name == "R":
            for tgt in inst.targets_copy():
                P[tgt.value] = 0
        else:
            P = P.after(inst)
    return flipped


LINEAR_RULES = {
    # (delay in rounds, stabilizer index of the control record, Pauli, qubit)
    "A_covariant_lookup": [(0, 0, "X", 0), (0, 1, "X", 2),
                           (0, 2, "Z", 0), (0, 3, "Z", 3)],
    # The other member of every ambiguous pair (non-covariant, differs from
    # the implementer's A' = X3, X2, Z0, Z4).
    "L_other_pair": [(0, 0, "X", 3), (0, 1, "X", 4), (0, 2, "Z", 2),
                     (0, 3, "Z", 4)],
    "L_Y": [(0, 0, "Y", 0), (0, 3, "Y", 4), (0, 1, "Y", 2), (0, 2, "Z", 1)],
    "L_delayed": [(1, 2, "Z", 2), (1, 1, "X", 4), (0, 3, "Z", 4),
                  (1, 0, "Y", 3)],
}


class FrameModel:
    """Everything needed to compare physical feedback and frame tracking."""

    def __init__(self, seq, depth, p):
        lc = Q.build_lrb_circuit(seq, depth, p, terminal="data5")
        self.insts, self.rounds, self.n_meas, self.data_rec = \
            round_structure(lc.circuit)
        lc0 = Q.build_lrb_circuit(seq, depth, 0.0, terminal="data5",
                                  noise=None)
        self.insts0, rounds0, n0, data0 = round_structure(lc0.circuit)
        assert [r for _, r in rounds0] == [r for _, r in self.rounds]
        assert n0 == self.n_meas and data0 == self.data_rec
        self.rounds0 = rounds0
        self.obs_recs = [self.data_rec[0], self.data_rec[3]]
        self._cache = {}

    def effect(self, k: int, pauli: str, qubit: int) -> set:
        key = (k, pauli, qubit)
        if key not in self._cache:
            i0, recs = self.rounds0[k]
            self._cache[key] = pauli_flips(self.insts0, i0, recs[3] + 1,
                                           pauli, qubit)
        return self._cache[key]

    def plain(self, observable_recs) -> stim.Circuit:
        out = stim.Circuit()
        for inst in self.insts:
            out.append(inst)
        out.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(r - self.n_meas) for r in observable_recs],
                   0)
        return out

    def feedback(self, rule, noise_q: float = 0.0) -> stim.Circuit:
        out = stim.Circuit()
        at = {i: k for k, (i, _) in enumerate(self.rounds)}
        m = 0
        for i, inst in enumerate(self.insts):
            out.append(inst)
            if inst.name == "M":
                m += len(inst.targets_copy())
            if i in at:
                k = at[i]
                touched = []
                for delay, s, pauli, q in rule:
                    if k - delay < 0:
                        continue
                    rec = self.rounds[k - delay][1][s]
                    out.append("C" + pauli, [stim.target_rec(rec - m), q])
                    touched.append(q)
                if noise_q > 0 and touched:
                    out.append("DEPOLARIZE1", sorted(set(touched)), noise_q)
        out.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(r - self.n_meas) for r in self.obs_recs],
                   0)
        return out

    def frame_observable(self, rule) -> list[int]:
        """Records whose parity is the frame-corrected logical output.

        The controller's view a_j of control record j is the measured record
        XOR the flips the frame (earlier decisions) causes on it; a_j is
        kept as a GF(2) combination of measured records (python int mask)."""
        a: dict[int, int] = {}
        applied = []     # (insertion round, control record, flipped set)

        def view(j, round_j):
            v = 1 << j
            for k, ctrl, flipped in applied:
                if k < round_j and j in flipped:
                    v ^= a[ctrl]
            return v

        for k in range(len(self.rounds)):
            for delay, s, pauli, q in rule:
                if k - delay < 0:
                    continue
                j = self.rounds[k - delay][1][s]
                if j not in a:
                    a[j] = view(j, k - delay)
                applied.append((k, j, self.effect(k, pauli, q)))
        out = 0
        for r in self.obs_recs:
            out ^= 1 << r
        for k, ctrl, flipped in applied:
            if sum(r in flipped for r in self.obs_recs) % 2:
                out ^= a[ctrl]
        return [r for r in range(self.n_meas) if (out >> r) & 1]

    def segments(self):
        """Noisy circuit cut after every check round (for FlipSimulator)."""
        cuts = [i for i, _ in self.rounds]
        segs, cur = [], stim.Circuit()
        for i, inst in enumerate(self.insts):
            cur.append(inst)
            if i in cuts:
                segs.append(cur)
                cur = stim.Circuit()
        segs.append(cur)
        return segs


def flip_probability(circuit: stim.Circuit) -> float:
    """Exact P(observable 0 flips): XOR of independent DEM mechanisms."""
    prod = 1.0
    for inst in circuit.detector_error_model().flattened():
        if inst.type != "error":
            continue
        if sum(t.is_logical_observable_id() and t.val == 0
               for t in inst.targets_copy()) % 2:
            prod *= 1.0 - 2.0 * inst.args_copy()[0]
    return 0.5 * (1.0 - prod)


# Non-linear rules for FlipSimulator --------------------------------------

def rule_and_q1(k, view, state):
    """The suggested rule: (1,1) on the X checks -> Z1; (1,1) on the Z
    checks -> X1. Every logical gate maps qubit 1 to itself, so these
    corrections never reach the X0 X3 readout: logically inert, but they
    change every later syndrome (tests the record-level identity)."""
    return {("Z", 1): view[:, 2] & view[:, 3], ("X", 1): view[:, 0] & view[:, 1]}


def rule_and_q0q3(k, view, state):
    """Logically active AND rule: (1,1) on the X checks -> Z0 (flips the X0
    readout); (1,1) on the Z checks -> X3; (1,0) on the X checks -> Z2."""
    return {("Z", 0): view[:, 2] & view[:, 3],
            ("X", 3): view[:, 0] & view[:, 1],
            ("Z", 2): view[:, 2] & ~view[:, 3]}


def rule_and_two_round(k, view, state):
    """With memory: Z0 only if the X syndrome is (1,1) now AND was (1,1) in
    the previous round's controller view; X3 likewise for the Z checks."""
    prev = state.get("prev")
    now_x = view[:, 2] & view[:, 3]
    now_z = view[:, 0] & view[:, 1]
    if prev is None:
        dec = {("Z", 0): np.zeros_like(now_x), ("X", 3): np.zeros_like(now_z)}
    else:
        dec = {("Z", 0): now_x & prev[0], ("X", 3): now_z & prev[1]}
    state["prev"] = (now_x, now_z)
    return dec


NONLINEAR_RULES = {"AND_Z1_X1_suggested": rule_and_q1,
                   "AND_Z0_X3_Z2": rule_and_q0q3,
                   "AND_two_round_Z0_X3": rule_and_two_round}


def run_flipsim(model: FrameModel, rule, shots: int, seed: int, mode: str):
    segs = model.segments()
    sim = stim.FlipSimulator(batch_size=shots, num_qubits=NUM_QUBITS,
                             seed=seed)
    frame_flip = np.zeros((model.n_meas, shots), dtype=bool)
    state: dict = {}
    decisions = []
    for k in range(len(model.rounds)):
        sim.do(segs[k])
        flips = sim.get_measurement_flips()
        recs = model.rounds[k][1]
        view = flips[recs, :].T.copy()
        if mode == "frame":
            view ^= frame_flip[recs, :].T
        dec = rule(k, view, state)
        decisions.append({key: v.copy() for key, v in dec.items()})
        for (pauli, q), mask in dec.items():
            if mode == "physical":
                full = np.zeros((NUM_QUBITS, shots), dtype=bool)
                full[q] = mask
                sim.broadcast_pauli_errors(pauli=pauli, mask=full)
            else:
                idx = sorted(model.effect(k, pauli, q))
                frame_flip[idx, :] ^= mask[None, :]
    sim.do(segs[-1])
    rec = sim.get_measurement_flips()
    x0, x3 = model.obs_recs
    raw_out = rec[x0] ^ rec[x3]
    out = raw_out ^ frame_flip[x0] ^ frame_flip[x3]
    return {"out": out, "raw_out": raw_out, "records": rec,
            "frame_flip": frame_flip, "decisions": decisions}


def cmd_c10(args) -> int:
    out = Path(args.out)
    probs = [float(x) for x in args.p.split(",")]
    depths = [int(d) for d in args.depths.split(",")]
    seqs = new_sequences(args.num_seq)
    rows = []
    max_diff = 0.0
    t0 = time.perf_counter()
    for p in probs:
        for i, seq in enumerate(seqs):
            for depth in depths:
                model = FrameModel(seq, depth, p)
                none = flip_probability(model.plain(model.obs_recs))
                for name, rule in LINEAR_RULES.items():
                    phys = flip_probability(model.feedback(rule))
                    frame = flip_probability(
                        model.plain(model.frame_observable(rule)))
                    noisy = flip_probability(model.feedback(rule, 0.1 * p))
                    noisy_p = flip_probability(model.feedback(rule, p))
                    max_diff = max(max_diff, abs(phys - frame))
                    rows.append({"p": p, "seq": i, "depth": depth,
                                 "rule": name, "P_flip_physical": phys,
                                 "P_flip_frame": frame,
                                 "abs_diff": abs(phys - frame),
                                 "P_flip_no_correction": none,
                                 "dP_noisy_gate_q=0.1p": noisy - frame,
                                 "dP_noisy_gate_q=p": noisy_p - frame})
        print(f"c10 exact p={p} done ({time.perf_counter() - t0:.0f}s), "
              f"max |dP| so far {max_diff:.3g}", flush=True)
    write_rows(out / "c10_exact_linear.csv", rows)

    # Cross-check of the implementer's rule-A numbers on study sequences.
    study = E.clifford_sequences(30, 22, STUDY_SEED)
    ref = {}
    ref_path = ROOT / ("studies/correction-decoders/qubit/results/D/"
                       "frame_rule_a_exact.csv")
    with open(ref_path) as handle:
        for row in csv.DictReader(handle):
            ref[(row["rule"], float(row["p"]), int(row["seq"]),
                 int(row["depth"]))] = row
    cross = []
    for p in (5e-4, 2e-3, 5e-3):
        for s in (0, 1, 2):
            for depth in (2, 22):
                model = FrameModel(study[s], depth, p)
                rule = LINEAR_RULES["A_covariant_lookup"]
                mine = {"P_flip_physical_feedback":
                        flip_probability(model.feedback(rule)),
                        "P_flip_frame": flip_probability(
                            model.plain(model.frame_observable(rule))),
                        "P_flip_no_correction": flip_probability(
                            model.plain(model.obs_recs)),
                        "P_flip_physical_noisy_q=0.1p": flip_probability(
                            model.feedback(rule, 0.1 * p)),
                        "P_flip_physical_noisy_q=p": flip_probability(
                            model.feedback(rule, p))}
                theirs = ref[("A", p, s, depth)]
                row = {"p": p, "seq": s, "depth": depth}
                for key, val in mine.items():
                    row[f"mine_{key}"] = val
                    row[f"impl_{key}"] = float(theirs[key])
                    row[f"absdiff_{key}"] = abs(val - float(theirs[key]))
                cross.append(row)
    write_rows(out / "c10_crosscheck_ruleA_vs_impl_csv.csv", cross)

    # Non-linear rules, identical noise, shot by shot.
    flip_rows = []
    for p in probs:
        for i, seq in enumerate(seqs):
            for depth in depths:
                if depth == 0:
                    continue
                model = FrameModel(seq, depth, p)
                for name, rule in NONLINEAR_RULES.items():
                    seed = 700_000 + 1000 * i + 17 * depth + int(p * 1e6)
                    ph = run_flipsim(model, rule, args.shots, seed, "physical")
                    fr = run_flipsim(model, rule, args.shots, seed, "frame")
                    dec_same = all(
                        np.array_equal(a[key], b[key])
                        for a, b in zip(ph["decisions"], fr["decisions"])
                        for key in a)
                    rec_identity = bool(np.array_equal(
                        ph["records"], fr["records"] ^ fr["frame_flip"]))
                    fired = np.zeros(args.shots, dtype=bool)
                    for d in ph["decisions"]:
                        for v in d.values():
                            fired |= v
                    measured_diff = int((ph["records"] != fr["records"]
                                         ).any(axis=1).sum())
                    flip_rows.append({
                        "p": p, "seq": i, "depth": depth, "rule": name,
                        "shots": args.shots,
                        "output_mismatches": int((ph["out"] != fr["out"]).sum()),
                        "decisions_identical": dec_same,
                        "all_records_phys_eq_frame_xor_frameflip": rec_identity,
                        "shots_with_correction": int(fired.sum()),
                        "record_rows_measured_differently": measured_diff,
                        "P_flip_physical": float(ph["out"].mean()),
                        "P_flip_frame": float(fr["out"].mean()),
                        "P_flip_no_correction_same_noise":
                            float(fr["raw_out"].mean())})
            print(f"c10 flipsim p={p} seq={i} done "
                  f"({time.perf_counter() - t0:.0f}s)", flush=True)
    write_rows(out / "c10_flipsim_nonlinear.csv", flip_rows)
    total = sum(r["shots"] for r in flip_rows)
    mism = sum(r["output_mismatches"] for r in flip_rows)
    print(f"linear exact: max |dP| = {max_diff:.3g} over {len(rows)} "
          f"comparisons; nonlinear: {mism} mismatches in {total} shots; "
          f"record identity in all runs: "
          f"{all(r['all_records_phys_eq_frame_xor_frameflip'] for r in flip_rows)}",
          flush=True)
    return 0


# ---------------------------------------------------------------------------
# c3: filter versus smoother of the logical frame, via the lead's MLD
# ---------------------------------------------------------------------------

def _dem_text(lines, num_detectors):
    parts = []
    for prob, dets, obs in lines:
        if not dets and not obs:
            continue
        tgt = " ".join(f"D{d}" for d in dets)
        if obs:
            tgt = (tgt + " L0").strip()
        parts.append(f"error({prob!r}) {tgt}")
    parts.append(f"detector D{num_detectors - 1}")
    return stim.DetectorErrorModel("\n".join(parts))


def _c3_task(args):
    seq_index, seq, p, depth, shots, seed, lags = args
    lc = Q.build_lrb_circuit(seq, depth, p, terminal="data5")
    slots = slots_from_coords(lc.circuit)
    dem = lc.circuit.detector_error_model()
    mechs = parse_dem(dem)
    u = latest_slots(mechs, slots)
    dets, obs, errs = dem.compile_sampler(seed=seed).sample(
        shots, return_errors=True)
    assert errs.shape[1] == len(mechs)
    # Ground truth: rebuild detectors/observable from the error columns.
    e64 = errs.astype(np.int64)
    flips_obs = np.array([o for _, _, o in mechs], dtype=np.int64)
    assert np.array_equal((e64 @ flips_obs) % 2 == 1, obs[:, 0])
    T = int(slots.max()) + 1
    l_true = np.zeros((shots, T), dtype=bool)
    for t in range(T):
        sel = (u <= t) & (flips_obs == 1)
        l_true[:, t] = (e64[:, sel].sum(axis=1) % 2) == 1
    names = ["smooth"] + [f"lag_w{w}" for w in lags]
    errors = np.zeros((len(names), T), dtype=np.int64)
    for t in range(T):
        lines = [(prob, dets_k, int(o and u[k] <= t))
                 for k, (prob, dets_k, o) in enumerate(mechs)]
        mld = M.TransferMatrixMLD(_dem_text(lines, len(slots)), slots)
        est = mld.log_likelihood_ratio(dets) < 0
        errors[0, t] = int((est != l_true[:, t]).sum())
        for j, w in enumerate(lags, 1):
            end = min(t + w, T - 1)
            keep = slots <= end
            new_index = -np.ones(len(slots), dtype=np.int64)
            new_index[keep] = np.arange(int(keep.sum()))
            sub = []
            for k, (prob, dets_k, o) in enumerate(mechs):
                if u[k] > end + 1:
                    continue
                ds = tuple(int(new_index[d]) for d in dets_k if keep[d])
                sub.append((prob, ds, int(o and u[k] <= t)))
            mld = M.TransferMatrixMLD(_dem_text(sub, int(keep.sum())),
                                      slots[keep])
            est = mld.log_likelihood_ratio(dets[:, keep]) < 0
            errors[j, t] = int((est != l_true[:, t]).sum())
    return p, seq_index, names, errors


def cmd_c3(args) -> int:
    probs = [float(x) for x in args.p.split(",")]
    seqs = new_sequences(args.num_seq)
    lags = [0, 1]
    tasks = [(i, seqs[i], p, args.depth, args.shots,
              880_000 + 1000 * i + int(p * 1e6), lags)
             for p in probs for i in range(args.num_seq)]
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for fut in as_completed([pool.submit(_c3_task, t) for t in tasks]):
            p, i, names, errors = fut.result()
            results[(p, i)] = errors
            print(f"c3 p={p} seq={i} done", flush=True)
    rng = np.random.default_rng(4321)
    slots_fit = np.arange(2, args.depth - 1)
    rows = []
    for p in probs:
        errs = np.stack([results[(p, i)] for i in range(args.num_seq)])

        def bulk(idx):
            e = errs[idx].sum(axis=0) / (len(idx) * args.shots)
            fid = 1 - 2 * e[:, slots_fit]
            return np.array([-np.polyfit(slots_fit, np.log(f), 1)[0]
                             for f in fid])

        point = bulk(np.arange(args.num_seq))
        boots = np.array([bulk(rng.integers(0, args.num_seq, args.num_seq))
                          for _ in range(400)])
        for k, name in enumerate(names):
            ratio = boots[:, k] / boots[:, 0]
            rows.append({"p": p, "estimator": name, "bulk_rate": point[k],
                         "bulk_rate_se_boot": boots[:, k].std(ddof=1),
                         "ratio_vs_smooth": point[k] / point[0],
                         "ratio_se_paired_boot": ratio.std(ddof=1),
                         "frame_error_slot10": errs[:, k, min(10, errs.shape[2] - 1)].sum()
                         / (args.num_seq * args.shots),
                         "final_error": errs[:, k, -1].sum()
                         / (args.num_seq * args.shots)})
        np.savez_compressed(Path(args.out) / f"c3_p{p:g}.npz", errors=errs,
                            names=np.array(names), shots=args.shots)
    write_rows(Path(args.out) / "c3_filter_vs_smoother.csv", rows)
    return 0


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def fit_offset0(depths, mean, std):
    """F = b f^m, weights 1/std^2 (floored), points with F > 1 dropped.

    b has a closed form for fixed f; f is scanned on a dense grid and refined
    with a bounded scalar minimiser. Written independently of the repo's
    grid-scan code (same objective)."""
    from scipy.optimize import minimize_scalar
    x = np.asarray(depths, float)
    y = np.asarray(mean, float)
    s = np.asarray(std, float)
    ok = np.isfinite(y) & (y <= 1.0)
    x, y, s = x[ok], y[ok], s[ok]
    pos = s[s > 0]
    floor = max(float(np.median(pos)) * 1e-3 if len(pos) else 1e-12, 1e-12)
    wts = 1.0 / np.where(s > 0, s, floor) ** 2

    def chi2(f):
        z = f ** x
        b = (wts * z * y).sum() / (wts * z * z).sum()
        return float((wts * (y - b * z) ** 2).sum()), b

    grid = np.linspace(0.3, 1.0, 7001)
    z = grid[None, :] ** x[:, None]
    bg = (wts[:, None] * z * y[:, None]).sum(axis=0) / (
        wts[:, None] * z * z).sum(axis=0)
    vals = (wts[:, None] * (y[:, None] - bg[None, :] * z) ** 2).sum(axis=0)
    i = int(np.argmin(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    res = minimize_scalar(lambda f: chi2(f)[0], bounds=(lo, hi),
                          method="bounded", options={"xatol": 1e-11})
    f = float(res.x)
    c2, b = chi2(f)
    return f, b, c2


def cmd_report(args) -> int:
    out = Path(args.out)
    summary = {}
    for tag in ("c1", "c1_study"):
        _report_c1(args, out, tag, summary)
    with open(out / f"summary_p{args.p:g}.json", "w") as handle:
        json.dump(summary, handle, indent=1, default=float)
    print(json.dumps(summary, indent=1, default=float)[:6000])
    return 0


def _report_c1(args, out, tag, summary):
    path = out / f"{tag}_p{args.p:g}.json"
    if path.exists():
        data = json.load(open(path))
        meta, results = data["meta"], data["results"]
        depths = meta["depths"]
        n_seq = meta["num_seq"]
        names = meta["decoders"]
        err = np.zeros((len(names), n_seq, len(depths)))
        agree_tot = {}
        info_tot = {}
        n10 = np.zeros((len(names), len(depths)))
        n01 = np.zeros((len(names), len(depths)))
        for res in results:
            d = depths.index(res["depth"])
            for k, name in enumerate(names):
                err[k, res["seq"], d] = res["errors"][name] / res["shots"]
                n10[k, d] += res["n10"][name]
                n01[k, d] += res["n01"][name]
            for key, val in res["agree"].items():
                if key.startswith("max_rel_gap"):
                    agree_tot[key] = max(agree_tot.get(key, 0.0), val)
                else:
                    agree_tot[key] = agree_tot.get(key, 0) + val
            for key, val in res["info"].items():
                info_tot[key] = info_tot.get(key, 0) + val
        shots = meta["shots"]
        fid = np.abs(1 - 2 * err)                     # (dec, seq, depth)
        rows = []
        fits = {}
        for k, name in enumerate(names):
            fits[name] = fit_offset0(depths, fid[k].mean(axis=0),
                                     fid[k].std(axis=0))
        off = names.index("offline")

        def ratio_for(idx, mask=None):
            dsel = np.ones(len(depths), bool) if mask is None else mask
            dd = np.asarray(depths)[dsel]
            fo = fit_offset0(dd, fid[off][idx][:, dsel].mean(axis=0),
                             fid[off][idx][:, dsel].std(axis=0))[0]
            outr = {}
            for k, name in enumerate(names):
                fk = fit_offset0(dd, fid[k][idx][:, dsel].mean(axis=0),
                                 fid[k][idx][:, dsel].std(axis=0))[0]
                outr[name] = (1 - fk) / (1 - fo)
            return outr

        rng = np.random.default_rng(99)
        point = ratio_for(np.arange(n_seq))
        boots = [ratio_for(rng.integers(0, n_seq, n_seq))
                 for _ in range(args.boot)]
        jack = [ratio_for(np.delete(np.arange(n_seq), j))
                for j in range(n_seq)]
        repo9 = np.isin(depths, [0, 2, 4, 6, 10, 14, 18, 20, 22])
        point9 = ratio_for(np.arange(n_seq), repo9)
        # Robustness: unweighted log-linear slope of mean F over m = 2..12.
        lin_mask = (np.asarray(depths) >= 2) & (np.asarray(depths) <= 12)
        slopes = {}
        for k, name in enumerate(names):
            fm = fid[k].mean(axis=0)[lin_mask]
            slopes[name] = -np.polyfit(np.asarray(depths)[lin_mask],
                                       np.log(fm), 1)[0]
        # Per-sequence fits (binomial weights).
        per_seq = {}
        for k, name in enumerate(names):
            vals = []
            for s in range(n_seq):
                e = err[k, s]
                se = 2 * np.sqrt(np.maximum(e * (1 - e), 1e-12) / shots)
                vals.append(fit_offset0(depths, fid[k, s], se)[0])
            per_seq[name] = vals
        claims = {"impl_w0": 1.188, "impl_w1": 1.086, "mine_w0": 1.188,
                  "mine_w1": 1.086, "impl_w4": 1.041, "impl_winf": 1.040}
        for k, name in enumerate(names):
            f, b, c2 = fits[name]
            bvals = np.array([bt[name] for bt in boots])
            jvals = np.array([jk[name] for jk in jack])
            jse = float(np.sqrt((n_seq - 1) / n_seq
                                * ((jvals - jvals.mean()) ** 2).sum()))
            seq_ratio = [(1 - a) / (1 - o) for a, o in
                         zip(per_seq[name], per_seq["offline"])]
            rows.append({
                "p": meta["p"], "decoder": name, "f": f, "b": b,
                "chi2_weighted": c2,
                "ratio_vs_offline": point[name],
                "ratio_se_boot_paired": float(bvals.std(ddof=1)),
                "ratio_se_jackknife": jse,
                "ratio_repo9_depths": point9[name],
                "ratio_loglinear_m2_12": slopes[name] / slopes["offline"],
                "per_seq_ratio_min": float(min(seq_ratio)),
                "per_seq_ratio_max": float(max(seq_ratio)),
                "claimed_ratio_C1": claims.get(name, ""),
                "mcnemar_excess_m22": (n10[k, -1] - n01[k, -1])
                / (n_seq * shots)})
        write_rows(out / f"{tag}_fits_p{args.p:g}.csv", rows)
        # Tie-rule sensitivity: paired (same shots, same bootstrap
        # resamples) ratio differences against my l0 rule, per-sequence
        # ratios, and pooled excess error over offline at small depths.
        tie_rows = []
        for w in (0, 1):
            ref = _mine_name(w, "l0")
            for name in [n for n in names if n.startswith(f"mine_w{w}")
                         or n == f"impl_w{w}"]:
                dvals = np.array([bt[name] - bt[ref] for bt in boots])
                seq_ratio = np.array([(1 - a) / (1 - o) for a, o in
                                      zip(per_seq[name], per_seq["offline"])])
                k = names.index(name)
                excess = {}
                for depth in (1, 2, 4, 6, 10):
                    d = depths.index(depth)
                    total = n_seq * shots
                    excess[f"excess_err_m{depth}"] = (
                        (n10[k, d] - n01[k, d]) / total)
                    excess[f"excess_err_m{depth}_se"] = (
                        np.sqrt(n10[k, d] + n01[k, d]) / total)
                tie_rows.append({
                    "lookahead": w, "decoder": name,
                    "pooled_ratio": point[name],
                    "pooled_ratio_minus_l0": point[name] - point[ref],
                    "paired_boot_se": float(dvals.std(ddof=1)),
                    "per_seq_ratios": " ".join(f"{v:.4f}" for v in seq_ratio),
                    "per_seq_mean": float(seq_ratio.mean()),
                    "per_seq_se": float(seq_ratio.std(ddof=1)
                                        / np.sqrt(n_seq)),
                    **excess})
        write_rows(out / f"{tag}_tie_sensitivity_p{args.p:g}.csv", tie_rows)
        summary[f"{tag}_ties"] = tie_rows
        e_rows = []
        for k, name in enumerate(names):
            for d, depth in enumerate(depths):
                e_rows.append({"decoder": name, "depth": depth,
                               "error_mean": err[k, :, d].mean(),
                               "error_se": err[k, :, d].std(ddof=1)
                               / np.sqrt(n_seq),
                               "F_mean": fid[k, :, d].mean(),
                               "n10_vs_offline": int(n10[k, d]),
                               "n01_vs_offline": int(n01[k, d])})
        write_rows(out / f"{tag}_errors_p{args.p:g}.csv", e_rows)
        summary[tag] = {"agreement": agree_tot, "info": info_tot,
                         "total_shot_circuits": n_seq * len(depths) * shots,
                         "rows": rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tamper")
    t.add_argument("--out", default=str(OUT))
    t.add_argument("--p", type=float, default=5e-3)
    t.add_argument("--shots", type=int, default=2000)
    c = sub.add_parser("c1")
    c.add_argument("--out", default=str(OUT))
    c.add_argument("--p", type=float, default=5e-3)
    c.add_argument("--num-seq", type=int, default=5)
    c.add_argument("--shots", type=int, default=40_000)
    c.add_argument("--depths", default="all")
    c.add_argument("--workers", type=int, default=4)
    c.add_argument("--study", action="store_true",
                   help="use the implementer's 30 study sequences instead")
    f = sub.add_parser("c10")
    f.add_argument("--out", default=str(OUT))
    f.add_argument("--p", default="0.0005,0.002,0.005")
    f.add_argument("--depths", default="0,1,2,6,10,22")
    f.add_argument("--num-seq", type=int, default=5)
    f.add_argument("--shots", type=int, default=50_000)
    s = sub.add_parser("c3")
    s.add_argument("--out", default=str(OUT))
    s.add_argument("--p", default="0.0005,0.002,0.005")
    s.add_argument("--depth", type=int, default=22)
    s.add_argument("--num-seq", type=int, default=5)
    s.add_argument("--shots", type=int, default=20_000)
    s.add_argument("--workers", type=int, default=4)
    r = sub.add_parser("report")
    r.add_argument("--out", default=str(OUT))
    r.add_argument("--p", type=float, default=5e-3)
    r.add_argument("--boot", type=int, default=300)
    args = parser.parse_args()
    Path(args.out).mkdir(parents=True, exist_ok=True)
    return {"tamper": cmd_tamper, "c1": cmd_c1, "c10": cmd_c10,
            "c3": cmd_c3, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
