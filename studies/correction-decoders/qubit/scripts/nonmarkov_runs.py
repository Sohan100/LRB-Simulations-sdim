"""WP-F dense-depth runs: non-Markovianity of LRB-D versus LRB with correction.

Every method is scored on identical shots of the same circuit (one Stim
detector sample per (sequence, depth) cell), as in
``lrb_qubit.experiment.run_point``, whose seeding and bookkeeping this module
follows. Additions over ``run_point``:

``raw``
    No decoding and no rejection (the bare logical readout x0 + x3). Used as
    a control.
``mld5_<policy>``
    Information-restricted exact MLD: the transfer-matrix decoder sees only
    the raw syndromes of the check rounds that LRB-D ``<policy>`` uses, plus
    the two final data X-stabilizer detectors. It is compared with
    ``lrbd_<policy>_final`` on the same shots: same noise, same syndrome
    information, decoding instead of postselection.
``--variant invk``
    Circuits with an extra check round after every k-th inverse Clifford
    (``build_lrb_circuit(..., inverse_check_every=k)``). Methods there:
    ``lrbd_unif_1`` (inverse checks executed but not used), ``lrbd_sym``
    (all rounds used), their ``_final`` variants, ``mld5`` (all rounds),
    ``mld5_fwd`` (forward and terminal rounds only) and ``raw``.

Physical RB at ``p_RB = 5 p`` uses ``experiment.run_physical_rb`` unchanged.

Outputs (``--out``, default ``studies/correction-decoders/qubit/results/F``):
``<variant>_<ptag>.npz`` in the ``PointResult.to_npz`` layout (load with
``PointResult.from_npz``; MLD ``|LLR|`` histograms on ``GAP_EDGES``),
``rb_<ptag>.npz`` and ``<variant>_<ptag>.json`` (metadata and timing).

Usage (from the repository root, on the shared allocation; see ALLOCATION.md):

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --variant std --p 2e-3 \
        --shots 400000 --workers 24
    python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --variant inv1 --p 2e-3 \
        --shots 200000 --depths 0:30:2 --workers 24
    python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --selftest   # consistency checks
    python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --low-order --p 5e-4,2e-3,5e-3 \
        --workers 4                                         # exact expansion
    python studies/correction-decoders/qubit/scripts/nonmarkov_runs.py --exact --p 5e-4,2e-3,5e-3 \
        --workers 4                       # exact LRB-D and null-decoder curves
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Single-threaded BLAS in every process (see API.md): re-exec once if needed.
_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402
import stim  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

SEQ_SEED = 20261028
NUM_SEQ = 30
MAX_DEPTH = 30
POLICIES = (("unif", 1), ("unif", 2), ("unif", 4), ("const", 1),
            ("const", 4))
RESTRICTED = (("unif", 2), ("unif", 4), ("const", 1), ("const", 4))
DEFAULT_OUT = ROOT / "studies" / "correction-decoders" / "qubit" / "results" / "F"


# ---------------------------------------------------------------------------
# Fast raw syndromes and information-restricted detectors
# ---------------------------------------------------------------------------

def _subset_matrix(subsets: Sequence[Sequence[int]]) -> np.ndarray:
    """4x4 GF(2) matrix: row s has ones at the generators in ``subsets[s]``."""
    mat = np.zeros((4, 4), dtype=np.uint8)
    for s, subset in enumerate(subsets):
        for g in subset:
            mat[s, g] ^= 1
    return mat


def _lut(mat: np.ndarray) -> np.ndarray:
    """Lookup table v -> (mat @ v) mod 2 on 4-bit integers (bit s = stab s)."""
    table = np.zeros(16, dtype=np.uint8)
    for v in range(16):
        bits = np.array([(v >> s) & 1 for s in range(4)], dtype=np.uint8)
        out = (mat @ bits) % 2
        table[v] = int(sum(int(b) << s for s, b in enumerate(out)))
    return table


_WEIGHTS = (1 << np.arange(4)).astype(np.uint8)


def raw_int_syndromes(lc: Q.LRBCircuit, dets: np.ndarray) -> np.ndarray:
    """Raw per-round syndromes as 4-bit integers, shape (shots, rounds).

    Same content as ``LRBCircuit.raw_syndromes`` (bit s of entry [i, k] is
    ``raw[i, k, s]``), computed with one table lookup per round.
    """
    shots = dets.shape[0]
    out = np.zeros((shots, lc.num_rounds), dtype=np.uint8)
    for k, label in enumerate(lc.rounds):
        cur = (dets[:, 4 * k:4 * k + 4].astype(np.uint8) * _WEIGHTS).sum(
            axis=1, dtype=np.uint8)
        if k > 0:
            cur ^= _lut(_subset_matrix(lc.back[label]))[out[:, k - 1]]
        out[:, k] = cur
    return out


class RestrictedDecoder:
    """Exact MLD that sees only the raw syndromes of selected check rounds.

    The detectors are differences between consecutive *selected* rounds
    (back-propagated through the logical gates in between), plus, for
    ``data5`` circuits, the two final data X-stabilizer detectors. The DEM is
    built from a copy of the LRB circuit whose detector annotations are
    replaced; the noise and the logical observable are unchanged.
    """

    def __init__(self, lc: Q.LRBCircuit, positions: Sequence[int],
                 final: bool = True):
        positions = sorted(set(int(k) for k in positions))
        if positions[-1] != lc.num_rounds - 1:
            raise ValueError("The terminal round must be selected.")
        self.positions = positions
        self.final = final and lc.terminal == "data5"
        self.luts: list[Optional[np.ndarray]] = []
        circuit = stim.Circuit()
        for inst in lc.circuit.flattened():
            if inst.name not in ("DETECTOR", "OBSERVABLE_INCLUDE"):
                circuit.append(inst)
        n_meas = circuit.num_measurements
        n_round_meas = 4 * lc.num_rounds
        data_records = list(range(n_round_meas, n_meas))
        prev = None
        slots = []
        for i, k in enumerate(positions):
            if prev is None:
                subsets = tuple(() for _ in range(4))
                self.luts.append(None)
            else:
                ops = []
                for q in range(prev + 1, k + 1):
                    ops.extend(lc.segments[q])
                subsets = Q._back_subsets(ops, range(4))
                self.luts.append(_lut(_subset_matrix(subsets)))
            for s in range(4):
                targets = [4 * k + s]
                if prev is not None:
                    targets += [4 * prev + g for g in subsets[s]]
                circuit.append("DETECTOR", [stim.target_rec(r - n_meas)
                                            for r in targets], [s, k])
                slots.append(i)
            prev = k
        if self.final:
            # Same final detectors as build_lrb_circuit (data5).
            readout = dict(zip((0, 1, 2, 3, 4), data_records))
            fsubsets = Q._back_subsets(lc.segments[-1], (2, 3))
            last = positions[-1]
            for s, subset in zip((2, 3), fsubsets):
                support = [q for q, ch in enumerate(
                    Q.C.STABILIZER_PAULIS[s]) if ch != "_"]
                targets = [readout[q] for q in support]
                targets += [4 * last + g for g in subset]
                circuit.append("DETECTOR", [stim.target_rec(r - n_meas)
                                            for r in targets],
                               [s, lc.depth + 1])
                slots.append(len(positions))
        for inst in lc.circuit.flattened():
            if inst.name == "OBSERVABLE_INCLUDE":
                circuit.append(inst)
        self.circuit = circuit
        self.decoder = M.TransferMatrixMLD(circuit.detector_error_model(),
                                           np.asarray(slots))
        self.n_round_dets = 4 * lc.num_rounds

    def events(self, raw_int: np.ndarray, dets: np.ndarray) -> np.ndarray:
        """Restricted detection events from raw syndromes and full events."""
        cols = []
        prev = None
        for lut, k in zip(self.luts, self.positions):
            value = raw_int[:, k].copy()
            if prev is not None:
                value ^= lut[raw_int[:, prev]]
            cols.append(value)
            prev = k
        ints = np.stack(cols, axis=1)
        bits = ((ints[:, :, None] >> np.arange(4, dtype=np.uint8)) & 1)
        out = bits.reshape(ints.shape[0], -1).astype(bool)
        if self.final:
            out = np.concatenate([out, dets[:, self.n_round_dets:]], axis=1)
        return out

    def llr(self, raw_int: np.ndarray, dets: np.ndarray) -> np.ndarray:
        return self.decoder.log_likelihood_ratio(self.events(raw_int, dets))


def _hist(hist: np.ndarray, gap: np.ndarray, wrong: np.ndarray) -> None:
    """Same binning as ``experiment._histogram_gap``."""
    bins = np.clip(np.searchsorted(E.GAP_EDGES, gap, side="right") - 1, 0,
                   len(E.GAP_EDGES) - 2)
    np.add.at(hist, (bins, wrong.astype(np.int64)), 1)


def _positions(lc: Q.LRBCircuit, labels) -> list[int]:
    return [lc.rounds.index(r) for r in labels]


# ---------------------------------------------------------------------------
# One (sequence, depth) cell
# ---------------------------------------------------------------------------

def run_cell(variant: str, p: float, sequence: Sequence[str], depth: int,
             shots: int, seed: int, batch_size: int = 100_000) -> dict:
    """All methods of ``variant`` on identical shots of one circuit.

    Returns ``{method: (shots, accepted, errors)}`` and, under the key
    ``"__hist__"``, ``{method: |LLR| histogram (bins, 2)}``.
    """
    counts: dict[str, list[int]] = {}
    hists: dict[str, np.ndarray] = {}

    def add(name, n, acc, err):
        c = counts.setdefault(name, [0, 0, 0])
        c[0] += int(n)
        c[1] += int(acc)
        c[2] += int(err)

    def hist(name):
        if name not in hists:
            hists[name] = np.zeros((len(E.GAP_EDGES) - 1, 2), np.int64)
        return hists[name]

    if variant == "std":
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
        lc2 = Q.build_lrb_circuit(sequence, depth, p, terminal="data2")
        mld5 = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                   M.lrb_detector_slots(lc))
        mld2 = M.TransferMatrixMLD(lc2.circuit.detector_error_model(),
                                   M.lrb_detector_slots(lc2))
        restricted = {}
        for policy, k in RESTRICTED:
            pos = _positions(lc, B.used_rounds(policy, k, depth))
            if len(pos) == lc.num_rounds:
                continue  # identical to mld5
            restricted[f"mld5_{policy}_{k}"] = RestrictedDecoder(lc, pos)
        policy_pos = {f"lrbd_{policy}_{k}": _positions(
            lc, B.used_rounds(policy, k, depth)) for policy, k in POLICIES}
    elif variant.startswith("inv"):
        every = int(variant[3:])
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5",
                                 inverse_check_every=every)
        mld5 = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                   M.lrb_detector_slots(lc))
        mld2 = None
        fwd = _positions(lc, B.used_rounds("unif", 1, depth))
        restricted = {}
        if len(fwd) < lc.num_rounds:
            restricted["mld5_fwd"] = RestrictedDecoder(lc, fwd)
        policy_pos = {"lrbd_unif_1": fwd,
                      "lrbd_sym": list(range(lc.num_rounds))}
    else:
        raise ValueError(f"Unknown variant {variant!r}")

    n_round = 4 * lc.num_rounds
    sampler = lc.circuit.compile_detector_sampler(seed=seed)
    done = 0
    while done < shots:
        n = min(batch_size, shots - done)
        dets, obs = sampler.sample(n, separate_observables=True)
        flip = obs[:, 0]
        raw = raw_int_syndromes(lc, dets)
        nonzero = raw != 0
        final_clean = ~dets[:, n_round:].any(axis=1)
        add("raw", n, n, flip.sum())
        for name, pos in policy_pos.items():
            acc = ~nonzero[:, pos].any(axis=1)
            add(name, n, acc.sum(), (acc & flip).sum())
            acc_f = acc & final_clean
            add(name + "_final", n, acc_f.sum(), (acc_f & flip).sum())
        decoders = [("mld5", lambda: mld5.log_likelihood_ratio(dets))]
        if mld2 is not None:
            decoders.append(("mld2", lambda: mld2.log_likelihood_ratio(
                dets[:, :n_round])))
        for name, dec in restricted.items():
            decoders.append((name, lambda dec=dec: dec.llr(raw, dets)))
        for name, fn in decoders:
            llr = fn()
            wrong = (llr < 0) != flip
            add(name, n, n, wrong.sum())
            _hist(hist(name), np.abs(llr), wrong)
        done += n
    if variant == "std":
        # Restricted decoders that coincide with mld5 (all rounds used).
        for policy, k in RESTRICTED:
            name = f"mld5_{policy}_{k}"
            if name not in restricted:
                counts[name] = list(counts["mld5"])
                hists[name] = hists["mld5"].copy()
        # const=0: separate circuit without check rounds (as in run_point).
        lc0 = Q.build_lrb_const0_circuit(sequence, depth, p)
        dec0 = M.TransferMatrixMLD(lc0.circuit.detector_error_model(),
                                   M.lrb_detector_slots(lc0))
        sampler0 = lc0.circuit.compile_detector_sampler(seed=seed + 7)
        done = 0
        while done < shots:
            n = min(batch_size, shots - done)
            dets, obs = sampler0.sample(n, separate_observables=True)
            flip = obs[:, 0]
            acc = ~dets.any(axis=1)
            add("const0", n, acc.sum(), (acc & flip).sum())
            llr = (dec0.log_likelihood_ratio(dets) if dets.shape[1] > 0
                   else np.full(n, np.inf))
            wrong = (llr < 0) != flip
            add("const0_mld", n, n, wrong.sum())
            _hist(hist("const0_mld"), np.abs(llr), wrong)
            done += n
    counts["__hist__"] = hists
    return counts


def _task(args):
    variant, p, seq_index, d_index, sequence, depth, shots, seed = args
    start = time.perf_counter()
    out = run_cell(variant, p, sequence, depth, shots, seed)
    return seq_index, d_index, out, time.perf_counter() - start


def cell_seed(base: int, p_tag_index: int, seq_index: int, d_index: int,
              variant: str) -> int:
    v = 0 if variant == "std" else 50 + int(variant[3:])
    return (base + 10_000_000 * v + 100_000 * p_tag_index + 1000 * seq_index
            + d_index)


def parse_depths(text: str) -> list[int]:
    if ":" in text:
        a, b, s = (int(x) for x in text.split(":"))
        return list(range(a, b + 1, s))
    return [int(x) for x in text.split(",")]


P_TAGS = {5e-4: "p5e-4", 2e-3: "p2e-3", 5e-3: "p5e-3"}
P_INDEX = {5e-4: 0, 2e-3: 1, 5e-3: 2}


def main_run(args) -> int:
    """One chunk of sequences (``--seq-range a:b``) for one ``p``.

    Counts of sequences outside the chunk stay zero, so chunk files merge by
    addition (``--merge``). Seeds depend only on (p, global sequence index,
    depth index), so chunking does not change any shot.
    """
    p = float(args.p)
    depths = parse_depths(args.depths)
    sequences = E.clifford_sequences(NUM_SEQ, MAX_DEPTH, SEQ_SEED)
    a, b = (int(x) for x in args.seq_range.split(":"))
    chunk = list(range(a, min(b, len(sequences))))
    tag = P_TAGS.get(p, f"p{p:g}")
    pidx = P_INDEX.get(p, 9)
    out = Path(args.out) / "chunks"
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{args.variant}_{tag}_s{a}-{b}"
    tasks = []
    for si in chunk:
        for di, depth in enumerate(depths):
            tasks.append((args.variant, p, si, di, sequences[si], depth,
                          args.shots,
                          cell_seed(args.seed, pidx, si, di, args.variant)))
    tasks.sort(key=lambda t: -t[5])  # longest first
    result = E.PointResult(p, depths, len(sequences))
    start = time.perf_counter()
    cell_time = np.zeros((len(sequences), len(depths)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_task, t) for t in tasks]
        for count, fut in enumerate(as_completed(futures), 1):
            si, di, cell, seconds = fut.result()
            cell_time[si, di] = seconds
            hists = cell.pop("__hist__")
            for name, (n, acc, err) in cell.items():
                result.table(name).add(si, di, n, acc, err)
            for name, h in hists.items():
                result.hist(name)[si, di] += h
            if count % max(1, len(tasks) // 20) == 0:
                print(f"[{time.perf_counter() - start:7.1f}s] {count}/"
                      f"{len(tasks)} cells", flush=True)
    result.to_npz(str(out / f"{stem}.npz"))
    meta = {"variant": args.variant, "p": p, "depths": depths,
            "sequences_done": chunk, "num_seq_total": len(sequences),
            "shots_per_cell": args.shots, "seq_seed": SEQ_SEED,
            "seed_base": args.seed,
            "wall_seconds": time.perf_counter() - start,
            "core_seconds": float(cell_time.sum()), "workers": args.workers,
            "methods": sorted(result.tables), "host": os.uname().nodename}
    (out / f"{stem}.json").write_text(json.dumps(meta, indent=1))
    print(f"wrote {out / (stem + '.npz')} in {meta['wall_seconds']:.0f}s "
          f"({meta['core_seconds']:.0f} core-s)", flush=True)
    return 0


def merge_run(args) -> int:
    """Merge chunk files of one (variant, p) and run physical RB for std."""
    p = float(args.p)
    tag = P_TAGS.get(p, f"p{p:g}")
    pidx = P_INDEX.get(p, 9)
    out = Path(args.out)
    parts = sorted(out.glob(f"chunks/{args.variant}_{tag}_s*.npz"))
    if not parts:
        raise SystemExit("no chunk files")
    merged = None
    done: list[int] = []
    metas = []
    for part in parts:
        meta = json.loads(part.with_suffix(".json").read_text())
        metas.append(meta)
        if set(done) & set(meta["sequences_done"]):
            raise SystemExit(f"overlapping chunks: {part}")
        done += meta["sequences_done"]
        pr = E.PointResult.from_npz(str(part))
        if merged is None:
            merged = pr
            continue
        assert merged.depths == pr.depths
        for name, table in pr.tables.items():
            target = merged.table(name)
            target.shots += table.shots
            target.accepted += table.accepted
            target.errors += table.errors
        for name, hist in pr.gap_hist.items():
            merged.hist(name)[...] += hist
    n_total = metas[0]["num_seq_total"]
    missing = sorted(set(range(n_total)) - set(done))
    stem = f"{args.variant}_{tag}"
    merged.to_npz(str(out / f"{stem}.npz"))
    info = {"variant": args.variant, "p": p, "depths": merged.depths,
            "chunks": [m | {"file": str(pt.name)} for m, pt in
                       zip(metas, parts)],
            "sequences_missing": missing,
            "core_seconds": sum(m["core_seconds"] for m in metas),
            "sequences": E.clifford_sequences(NUM_SEQ, MAX_DEPTH, SEQ_SEED)}
    (out / f"{stem}.json").write_text(json.dumps(info, indent=1))
    print(f"merged {len(parts)} chunks into {stem}.npz; missing {missing}",
          flush=True)
    if args.variant == "std" and args.rb_shots > 0:
        sequences = E.clifford_sequences(NUM_SEQ, MAX_DEPTH, SEQ_SEED)
        rb = E.run_physical_rb(5.0 * p, sequences, merged.depths,
                               args.rb_shots,
                               seed=args.seed + 999 + 100_000 * pidx)
        np.savez_compressed(out / f"rb_{tag}.npz", p_rb=5.0 * p,
                            depths=np.array(merged.depths), shots=rb.shots,
                            accepted=rb.accepted, errors=rb.errors)
        print(f"wrote rb_{tag}.npz", flush=True)
    return 0


# ---------------------------------------------------------------------------
# Exact low-order fault expansion (no sampling)
# ---------------------------------------------------------------------------

def low_order_cell(p: float, sequence: Sequence[str], depth: int,
                   variant: str = "std") -> dict:
    """Leading-order logical failure weights from the DEM, per method.

    Every DEM mechanism ``e`` (probability ``p_e``, detectors ``D_e``,
    observable flip ``L_e``) is treated as the only fault in the shot.

    * Decoders (first order): ``S1 = sum_e p_e [decision(D_e) != L_e]``, with
      the exact MLD decision at this ``p``. ``F = 1 - 2 S1 + O(p^2)``.
      ``S1_late`` restricts the sum to mechanisms detected only by the
      terminal round and the final data detectors (this class holds every
      inverse-half fault).
    * LRB-D policies: a mechanism's signature is its raw-syndrome pattern on
      the used rounds (plus the final data detectors for ``_final``).
      ``S1u = sum_{sig=0, L=1} p_e`` (undetected logical faults) and
      ``S2u = sum_{sig != 0} q0(sig) q1(sig)`` (undetected logical pairs;
      ``q_l(sig)`` = total probability of mechanisms with that signature and
      ``L = l``), so ``1 - F = 2 (S1u + S2u) + O(p^3)`` on accepted shots.
      ``S2u_late`` keeps only pairs whose common signature is non-zero only
      in the terminal round and final data detectors.
    """
    out: dict[str, float] = {}
    if variant == "std":
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
    else:
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5",
                                 inverse_check_every=int(variant[3:]))
    dem = lc.circuit.detector_error_model()
    mech = M.dem_mechanisms(dem)
    slots = M.lrb_detector_slots(lc)
    n_det = dem.num_detectors
    D = np.zeros((len(mech), n_det), dtype=bool)
    for i, mm in enumerate(mech):
        D[i, list(mm.detectors)] = True
    pe = np.array([mm.probability for mm in mech], dtype=float)
    L = np.array([mm.flips_observable for mm in mech], dtype=bool)
    has = D.any(axis=1)
    first_slot = np.where(has, np.where(D, slots[None, :], 10 ** 6).min(
        axis=1), -1)
    late = has & (first_slot >= lc.num_rounds - 1)
    dec = M.TransferMatrixMLD(dem, slots)
    wrong = (dec.log_likelihood_ratio(D) < 0) != L
    out["mld5:S1"] = float((pe * wrong).sum())
    out["mld5:S1_late"] = float((pe * wrong * late).sum())
    out["raw:S1"] = float((pe * L).sum())
    out["raw:S1_late"] = float((pe * L * late).sum())
    raw = raw_int_syndromes(lc, D)
    final = D[:, 4 * lc.num_rounds:]
    if variant == "std":
        policies = {f"lrbd_{pol}_{k}": _positions(
            lc, B.used_rounds(pol, k, depth)) for pol, k in POLICIES}
        for pol, k in RESTRICTED:
            pos = _positions(lc, B.used_rounds(pol, k, depth))
            if len(pos) < lc.num_rounds:
                rd = RestrictedDecoder(lc, pos)
                w = (rd.llr(raw, D) < 0) != L
            else:
                w = wrong
            out[f"mld5_{pol}_{k}:S1"] = float((pe * w).sum())
            out[f"mld5_{pol}_{k}:S1_late"] = float((pe * w * late).sum())
    else:
        policies = {"lrbd_unif_1": _positions(
            lc, B.used_rounds("unif", 1, depth)),
            "lrbd_sym": list(range(lc.num_rounds))}
    term = lc.num_rounds - 1
    for name, pos in policies.items():
        for suffix, use_final in (("", False), ("_final", True)):
            sig = raw[:, pos]
            if use_final:
                sig = np.concatenate([sig, final.astype(np.uint8)], axis=1)
            zero = ~sig.any(axis=1)
            s1 = float((pe * (zero & L)).sum())
            rej = float((pe * ~zero).sum())
            keys = np.ascontiguousarray(sig[~zero]).view(
                np.dtype((np.void, sig.shape[1])))[:, 0]
            uniq, inv = np.unique(keys, return_inverse=True)
            q0 = np.bincount(inv, weights=pe[~zero] * ~L[~zero],
                             minlength=len(uniq))
            q1 = np.bincount(inv, weights=pe[~zero] * L[~zero],
                             minlength=len(uniq))
            # signature non-zero only in the terminal round (+ final dets)
            sub = sig[~zero]
            tpos = [i for i, k in enumerate(pos) if k == term]
            other = np.ones(sub.shape[1], bool)
            other[tpos] = False
            if use_final:
                other[-2:] = False
            late_key = ~sub[:, other].any(axis=1)
            late_u = np.zeros(len(uniq), bool)
            late_u[inv] = late_key
            out[f"{name}{suffix}:S1u"] = s1
            out[f"{name}{suffix}:S2u"] = float((q0 * q1).sum())
            out[f"{name}{suffix}:S2u_late"] = float((q0 * q1)[late_u].sum())
            out[f"{name}{suffix}:R1"] = rej
    if variant == "std":
        lc0 = Q.build_lrb_const0_circuit(sequence, depth, p)
        dem0 = lc0.circuit.detector_error_model()
        mech0 = M.dem_mechanisms(dem0)
        D0 = np.zeros((len(mech0), dem0.num_detectors), dtype=bool)
        for i, mm in enumerate(mech0):
            D0[i, list(mm.detectors)] = True
        pe0 = np.array([mm.probability for mm in mech0], dtype=float)
        L0 = np.array([mm.flips_observable for mm in mech0], dtype=bool)
        if dem0.num_detectors and len(mech0):
            d0 = M.TransferMatrixMLD(dem0, M.lrb_detector_slots(lc0))
            w0 = (d0.log_likelihood_ratio(D0) < 0) != L0
        else:
            w0 = L0
        out["const0_mld:S1"] = float((pe0 * w0).sum())
        zero = ~D0.any(axis=1)
        out["const0:S1u"] = float((pe0 * (zero & L0)).sum())
        keys = np.ascontiguousarray(D0[~zero]).view(
            np.dtype((np.void, D0.shape[1])))[:, 0]
        if len(keys):
            uniq, inv = np.unique(keys, return_inverse=True)
            q0 = np.bincount(inv, weights=pe0[~zero] * ~L0[~zero])
            q1 = np.bincount(inv, weights=pe0[~zero] * L0[~zero])
            out["const0:S2u"] = float((q0 * q1).sum())
        else:
            out["const0:S2u"] = 0.0
    return out


def exact_cell(p: float, sequence: Sequence[str], depth: int,
               variant: str = "std") -> dict:
    """Exact LRB-D and null-decoder statistics (lead's ``exact.py``).

    Returns ``{method: (P(accept), P(accept and logical flip))}`` for the
    LRB-D policies (with and without the final data check) and for ``null``
    (no decoding, no rejection: every slot marginalized).
    """
    from lrb_qubit import exact as X
    if variant == "std":
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5")
        policies = {f"lrbd_{pol}_{k}": _positions(
            lc, B.used_rounds(pol, k, depth)) for pol, k in POLICIES}
    else:
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5",
                                 inverse_check_every=int(variant[3:]))
        policies = {"lrbd_unif_1": _positions(
            lc, B.used_rounds("unif", 1, depth)),
            "lrbd_sym": list(range(lc.num_rounds))}
    dec = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                              M.lrb_detector_slots(lc))
    ev = X.RawPolicyEvaluator(lc, dec)
    out = {"null": X.exact_condition(dec, [])}
    for name, pos in policies.items():
        out[name] = ev.evaluate(pos, False)
        out[name + "_final"] = ev.evaluate(pos, True)
    return out


def _exact_task(args):
    p, si, di, seq, depth, variant = args
    return si, di, exact_cell(p, seq, depth, variant)


def exact_run(args) -> int:
    depths = parse_depths(args.depths)
    sequences = E.clifford_sequences(NUM_SEQ, MAX_DEPTH, SEQ_SEED)
    sequences = sequences[:args.num_seq]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in (float(x) for x in args.p.split(",")):
        tasks = [(p, si, di, seq, d, args.variant)
                 for si, seq in enumerate(sequences)
                 for di, d in enumerate(depths)]
        tasks.sort(key=lambda t: -t[4])
        acc: dict[str, np.ndarray] = {}
        flip: dict[str, np.ndarray] = {}
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for si, di, cell in pool.map(_exact_task, tasks, chunksize=4):
                for key, (a, f) in cell.items():
                    if key not in acc:
                        shape = (len(sequences), len(depths))
                        acc[key] = np.full(shape, np.nan)
                        flip[key] = np.full(shape, np.nan)
                    acc[key][si, di] = a
                    flip[key][si, di] = f
        tag = P_TAGS.get(p, f"p{p:g}")
        name = f"exact_{args.variant}_{tag}.npz"
        arrays = {f"{k}__accept": v for k, v in acc.items()}
        arrays.update({f"{k}__acc_flip": v for k, v in flip.items()})
        np.savez_compressed(out / name, p=p, depths=np.array(depths),
                            methods=np.array(sorted(acc)), **arrays)
        print(f"wrote {name} in {time.perf_counter() - start:.0f}s",
              flush=True)
    return 0


def _low_task(args):
    p, si, di, seq, depth, variant = args
    return si, di, low_order_cell(p, seq, depth, variant)


def low_order_run(args) -> int:
    depths = parse_depths(args.depths)
    # The first 30 sequences are the main ones; more (same RNG streams,
    # indices 30, 31, ...) can be added for the exact expansion.
    sequences = E.clifford_sequences(max(NUM_SEQ, args.num_seq), MAX_DEPTH,
                                     SEQ_SEED)[:args.num_seq]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for p in (float(x) for x in args.p.split(",")):
        tasks = [(p, si, di, seq, d, args.variant)
                 for si, seq in enumerate(sequences)
                 for di, d in enumerate(depths)]
        tasks.sort(key=lambda t: -t[4])
        arrays: dict[str, np.ndarray] = {}
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for si, di, cell in pool.map(_low_task, tasks, chunksize=4):
                for key, val in cell.items():
                    if key not in arrays:
                        arrays[key] = np.full((len(sequences), len(depths)),
                                              np.nan)
                    arrays[key][si, di] = val
        tag = P_TAGS.get(p, f"p{p:g}")
        name = f"loworder_{args.variant}_{tag}" + (
            f"_n{args.num_seq}" if args.num_seq != NUM_SEQ else "") + ".npz"
        np.savez_compressed(out / name, p=p, depths=np.array(depths),
                            keys=np.array(sorted(arrays)),
                            **{k.replace(":", "__"): v
                               for k, v in arrays.items()})
        print(f"wrote {name} in {time.perf_counter() - start:.0f}s",
              flush=True)
    return 0


# ---------------------------------------------------------------------------
# Self tests (run on a login node in a few seconds)
# ---------------------------------------------------------------------------

def selftest() -> int:
    seqs = E.clifford_sequences(3, 12, SEQ_SEED)
    p = 5e-3
    # 1. fast raw syndromes == LRBCircuit.raw_syndromes
    for depth, every in ((0, 0), (5, 0), (12, 0), (6, 1), (7, 2)):
        lc = Q.build_lrb_circuit(seqs[1], depth, p, terminal="data5",
                                 inverse_check_every=every)
        dets, _ = lc.circuit.compile_detector_sampler(seed=5).sample(
            4000, separate_observables=True)
        ref = lc.raw_syndromes(dets)
        mine = raw_int_syndromes(lc, dets)
        bits = ((mine[:, :, None] >> np.arange(4, dtype=np.uint8)) & 1
                ).astype(bool)
        assert (bits == ref).all(), (depth, every)
    print("ok raw_int_syndromes == LRBCircuit.raw_syndromes")
    # 2. restricted events == detectors of the restricted circuit
    for depth, every, labels in ((8, 0, B.used_rounds("unif", 2, 8)),
                                 (9, 0, B.used_rounds("unif", 4, 9)),
                                 (7, 0, B.used_rounds("const", 1, 7)),
                                 (10, 0, B.used_rounds("const", 4, 10)),
                                 (6, 1, B.used_rounds("unif", 1, 6))):
        lc = Q.build_lrb_circuit(seqs[2], depth, p, terminal="data5",
                                 inverse_check_every=every)
        pos = _positions(lc, labels)
        rd = RestrictedDecoder(lc, pos)
        meas = lc.circuit.compile_sampler(seed=9).sample(3000)
        full = lc.circuit.compile_m2d_converter().convert(
            measurements=meas, separate_observables=False)
        full_dets = full[:, :lc.circuit.num_detectors]
        restricted = rd.circuit.compile_m2d_converter().convert(
            measurements=meas, separate_observables=False)
        restricted = restricted[:, :rd.circuit.num_detectors]
        mine = rd.events(raw_int_syndromes(lc, full_dets), full_dets)
        assert (mine == restricted).all(), (depth, every, labels)
        # noiseless: restricted circuit is deterministic
        Q.build_lrb_circuit(seqs[2], depth, 0.0, terminal="data5")
    print("ok restricted detection events == restricted circuit detectors")
    # 3. restricted decoder with all rounds == mld5 (same LLRs)
    lc = Q.build_lrb_circuit(seqs[0], 6, p, terminal="data5")
    rd = RestrictedDecoder(lc, list(range(lc.num_rounds)))
    full = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                               M.lrb_detector_slots(lc))
    dets, _ = lc.circuit.compile_detector_sampler(seed=3).sample(
        5000, separate_observables=True)
    a = full.log_likelihood_ratio(dets)
    b = rd.llr(raw_int_syndromes(lc, dets), dets)
    assert np.allclose(a, b, atol=1e-8, rtol=1e-8), np.abs(a - b).max()
    print("ok restricted decoder with every round == mld5")
    # 4. run_cell reproduces run_point on identical shots
    for depth in (0, 4, 9):
        ref = E.run_point(p, [seqs[0]], [depth], 30000, const_checks=(0, 1, 4),
                          unif_checks=(1, 2, 4),
                          final_variant_policies=POLICIES, seed=777)
        cell = run_cell("std", p, seqs[0], depth, 30000, seed=777)
        for name, table in ref.tables.items():
            got = cell[name]
            exp = [int(table.shots[0, 0]), int(table.accepted[0, 0]),
                   int(table.errors[0, 0])]
            assert got == exp, (depth, name, got, exp)
        for name in ("mld5", "mld2", "const0_mld"):
            assert (cell["__hist__"][name] == ref.gap_hist[name][0, 0]).all()
    print("ok run_cell == experiment.run_point for all shared methods")
    # 5. noiseless sanity: every method perfect
    cell = run_cell("std", 0.0, seqs[0], 5, 2000, seed=1)
    cell.pop("__hist__")
    assert all(c[2] == 0 and c[1] == c[0] for c in cell.values()), cell
    cell = run_cell("inv1", 0.0, seqs[0], 5, 2000, seed=1)
    cell.pop("__hist__")
    assert all(c[2] == 0 and c[1] == c[0] for c in cell.values()), cell
    print("ok noiseless circuits: no errors, no rejection")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--variant", default="std",
                        help="std, or invK for inverse_check_every=K")
    parser.add_argument("--p", default="2e-3")
    parser.add_argument("--depths", default="0:30:1",
                        help="a:b:step or comma list")
    parser.add_argument("--num-seq", type=int, default=NUM_SEQ)
    parser.add_argument("--shots", type=int, default=400_000)
    parser.add_argument("--rb-shots", type=int, default=200_000)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4_000_000)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--seq-range", default="0:30",
                        help="chunk of global sequence indices a:b")
    parser.add_argument("--merge", action="store_true",
                        help="merge chunks/<variant>_<ptag>_s*.npz")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--exact", action="store_true",
                        help="exact LRB-D and null-decoder curves "
                             "(lrb_qubit.exact); --p may be a comma list")
    parser.add_argument("--low-order", action="store_true",
                        help="exact first/second-order fault expansion; "
                             "--p may be a comma list")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    if args.low_order:
        return low_order_run(args)
    if args.exact:
        return exact_run(args)
    if args.merge:
        return merge_run(args)
    return main_run(args)


if __name__ == "__main__":
    raise SystemExit(main())
