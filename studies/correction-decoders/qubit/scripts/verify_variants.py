"""Independent adversarial verification of work package H (schedules, ablations).

This script does NOT import ``lrb_qubit.variants`` except in the ``compare``
subcommand, which is meant to run last. It has its own circuit builder:

* noise emission written from the SI1000 description in CONTEXT.md
  (DEPOLARIZE1(0.1p) after 1q gates, DEPOLARIZE2(p) after 2q gates, each
  measurement layer: DEPOLARIZE1(2p) on every other wire, DEPOLARIZE1(p) and
  X_ERROR(5p) on each measured wire; reset layer: DEPOLARIZE1(2p) on every
  other wire, X_ERROR(2p) after each reset);
* check circuits written from CONTEXT.md, with three measurement schedules:
  ``serial`` (M5, M6, M7, M8 each its own layer), ``zx`` (M5 M6 after both Z
  checks, M7 M8 after both X checks) and ``parallel`` (all twelve CNOTs, then
  ONE layer measuring all four ancillas), plus ``parallel_xz`` (X-check CNOTs
  first; a robustness variant). ``-nr`` = measure without reset: the raw
  stabilizer value is then the XOR of the ancilla's current and previous
  records;
* two detector bases: ``raw`` (one detector per raw stabilizer value, plus the
  two final X parities) and ``diff`` (difference detectors, needed by the
  lead's transfer-matrix evaluators). The first-order census is invariant
  under the change of basis (it is an invertible GF(2) map on symptoms), which
  the self-test checks.

From the lead's package it uses only: the logical-gate decompositions and
encoder (``code``), the sequence generator (``experiment``), the reference
builder (``circuits``, for cross-checks), ``used_rounds`` (cross-checked
against an own implementation), ``mld.TransferMatrixMLD``,
``exact.RawPolicyEvaluator`` / ``exact_all_rounds`` and the repo fit
(``fits.fit_fixed_offset``).

Subcommands (outputs in ``--out``, default results/H-verify)::

    selftest   builder validation (determinism, equality with the lead's
               serial DEM, raw/diff basis invariance, evaluator vs brute force
               and vs sampled measurement records)
    census     first-order DEM census at p = 1e-6 (claims 2, 3, 5, 6, 9, 10, 13)
    census-summary  rebuild census_summary.json from the saved rows
    exact      shot-noise-free LRB-D with RawPolicyEvaluator (claims 11, 12, 13)
    paired     finite-p gain of NOISELESS checks, decoded on identical shots
               with and without the check detectors (claim 10)
    sample     finite-p decoded rates and penalty fractions (claim 6)
    compare    my circuits / numbers vs the implementer's variants.py and files

Commands used (repository root; allocation via srun --overlap --cpu-bind=none,
16 workers pinned with taskset)::

    python studies/correction-decoders/qubit/scripts/verify_variants.py selftest
    python studies/correction-decoders/qubit/scripts/verify_variants.py census --workers 16
    python studies/correction-decoders/qubit/scripts/verify_variants.py exact --workers 16 \
        --probs 5e-3,2e-3
    python studies/correction-decoders/qubit/scripts/verify_variants.py paired --workers 16 \
        --shots 300000 --probs 2e-3,5e-3,1e-2
    python studies/correction-decoders/qubit/scripts/verify_variants.py sample --workers 16 \
        --shots 100000 --probs 2e-3,5e-3
    python studies/correction-decoders/qubit/scripts/verify_variants.py compare
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

from collections import defaultdict  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import stim  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import exact as X  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import fits as FT  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

P_SMALL = 1e-6
DEPTHS = (0, 2, 4, 6, 10, 14, 18, 20, 22)
THEIR_SEED = 20261028           # the implementer's (and lead's) sequences
NEW_SEED = 777_000_001          # fresh sequences, never used by WP H
OUT_DEFAULT = ROOT / "studies/correction-decoders/qubit/results/H-verify"

# ---------------------------------------------------------------------------
# Code and check circuits (from CONTEXT.md)
# ---------------------------------------------------------------------------

NQ = 9
DATA = (0, 1, 2, 3, 4)
ANC = (5, 6, 7, 8)                     # ancilla of stabilizer s
STAB = ("ZZ_Z_", "_ZZ_Z", "XXX__", "_X_XX")   # S_Z1, S_Z2, S_X1, S_X2
X_STABS = (2, 3)
CHECK_GATES = (
    (("CX", 0, 5), ("CX", 1, 5), ("CX", 3, 5)),
    (("CX", 1, 6), ("CX", 2, 6), ("CX", 4, 6)),
    (("H", 7), ("CX", 7, 0), ("CX", 7, 1), ("CX", 7, 2), ("H", 7)),
    (("H", 8), ("CX", 8, 3), ("CX", 8, 1), ("CX", 8, 4), ("H", 8)),
)
# Each group: the gate blocks of these stabilizers (in this order), then one
# measurement layer of their ancillas.
SCHEDULES = {
    "serial": ((0,), (1,), (2,), (3,)),
    "zx": ((0, 1), (2, 3)),
    "parallel": ((0, 1, 2, 3),),
    "parallel_xz": ((2, 3, 0, 1),),
}


def support(s: int) -> list[int]:
    return [q for q, ch in enumerate(STAB[s]) if ch != "_"]


# ---------------------------------------------------------------------------
# Noise emission (own implementation of the repo's SI1000)
# ---------------------------------------------------------------------------

class Emitter:
    """Appends gates and SI1000 channels; tallies channel mass by category."""

    def __init__(self, p: float, check_scale: float = 1.0,
                 gate_scale: float = 1.0):
        self.c = stim.Circuit()
        self.p = p
        self.cs = check_scale
        self.gs = gate_scale
        self.nmeas = 0
        self.tally: dict[str, float] = defaultdict(float)

    def _ch(self, name, targets, prob, cat):
        targets = list(targets)
        if prob > 0 and targets:
            self.c.append(name, targets, prob)
            self.tally[cat] += prob * (len(targets) // (2 if name ==
                                                         "DEPOLARIZE2" else 1))

    def gate(self, name, targets, scale, cat):
        self.c.append(name, list(targets))
        if len(targets) == 1:
            self._ch("DEPOLARIZE1", targets, 0.1 * self.p * scale, cat + "_1q")
        else:
            self._ch("DEPOLARIZE2", targets, 1.0 * self.p * scale, cat + "_2q")

    def ideal(self, name, targets):
        self.c.append(name, list(targets))

    def measure_layer(self, qubits) -> dict[int, int]:
        s = self.cs * self.p
        others = [q for q in range(NQ) if q not in qubits]
        self._ch("DEPOLARIZE1", [q for q in others if q in DATA], 2.0 * s,
                 "meas_idle_data")
        self._ch("DEPOLARIZE1", [q for q in others if q not in DATA], 2.0 * s,
                 "meas_idle_anc")
        recs = {}
        for q in qubits:
            self._ch("DEPOLARIZE1", [q], 1.0 * s, "meas_depol")
            self._ch("X_ERROR", [q], 5.0 * s, "meas_flip")
            self.c.append("M", [q])
            recs[q] = self.nmeas
            self.nmeas += 1
        return recs

    def reset_layer(self, qubits):
        s = self.cs * self.p
        others = [q for q in range(NQ) if q not in qubits]
        self._ch("DEPOLARIZE1", others, 2.0 * s, "reset_idle")
        for q in qubits:
            self.c.append("R", [q])
            self._ch("X_ERROR", [q], 2.0 * s, "reset_flip")

    def ideal_measure(self, qubits) -> dict[int, int]:
        recs = {}
        for q in qubits:
            self.c.append("M", [q])
            recs[q] = self.nmeas
            self.nmeas += 1
        return recs


def check_round(em: Emitter, schedule: str, with_reset: bool) -> list[int]:
    """One check round; returns the record index of each stabilizer's ancilla."""
    recs: dict[int, int] = {}
    for group in SCHEDULES[schedule]:
        for s in group:
            for g in CHECK_GATES[s]:
                if g[0] == "H":
                    em.gate("H", (g[1],), em.cs, "check")
                else:
                    em.gate("CX", (g[1], g[2]), em.cs, "check")
        recs.update(em.measure_layer(sorted(ANC[s] for s in group)))
    if with_reset:
        em.reset_layer(ANC)
    return [recs[ANC[s]] for s in range(4)]


# ---------------------------------------------------------------------------
# Back-propagation of stabilizers (own implementation, stim PauliString.before)
# ---------------------------------------------------------------------------

_GEN = [stim.PauliString(s) for s in STAB]


def pull_back_subset(segment, s: int) -> tuple[int, ...]:
    """Generators whose product equals S_s pulled back through ``segment``.

    ``segment`` is a list of ideal data-qubit operations ``(gate, targets)``.
    The product is required to match including the sign (+), so a wrong sign
    convention in the logical gates would raise here.
    """
    circ = stim.Circuit()
    circ.append("I", list(DATA))
    for g, t in segment:
        circ.append(g, list(t))
    target = stim.PauliString(STAB[s]).before(circ)
    assert target.after(circ) == stim.PauliString(STAB[s])
    for mask in range(16):
        prod = stim.PauliString(5)
        for t in range(4):
            if (mask >> t) & 1:
                prod *= _GEN[t]
        if prod == target:
            return tuple(t for t in range(4) if (mask >> t) & 1)
    raise AssertionError(f"S_{s} pulled back is {target}, not a + stabilizer")


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

@dataclass
class MyCircuit:
    circuit: stim.Circuit
    depth: int
    rounds: list
    back: dict
    segments: list
    slots: np.ndarray
    n_round_dets: int
    round_records: dict
    data_records: dict
    reset: bool
    tally: dict

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)


def build(seq, depth: int, p: float, *, schedule: str = "serial",
          reset: bool = True, forward=None, inverse=(),
          terminal_check: bool = True, readout: str = "data5",
          detectors: str = "diff", check_scale: float = 1.0,
          gate_scale: float = 1.0) -> MyCircuit:
    """Own LRB circuit builder.

    forward: None = every forward round j = 0..depth-1, else the executed set.
    inverse: inverse-half rounds after inverse Clifford number j+1 (j+1 < depth).
    """
    em = Emitter(p, check_scale, gate_scale)
    for g, t in C.encoding_operations():
        em.ideal(g, t)
    fwd = set(range(depth)) if forward is None else {int(r) for r in forward}
    assert all(0 <= r < depth for r in fwd)
    inv = {int(j) for j in inverse}
    rounds, round_recs, segments, pending = [], {}, [], []

    def do_round(label, with_reset):
        round_recs[label] = check_round(em, schedule, with_reset and reset)
        rounds.append(label)
        segments.append(list(pending))
        pending.clear()

    for j in range(depth):
        ops = C.logical_clifford_operations(seq[j])
        for g, t in ops:
            em.gate(g, t, em.gs, "logical")
        pending.extend(ops)
        if j in fwd:
            do_round(j, True)
    label = 1000
    for j, desc in enumerate(reversed(list(seq[:depth]))):
        ops = C.logical_clifford_operations(desc, inverse=True)
        for g, t in ops:
            em.gate(g, t, em.gs, "logical")
        pending.extend(ops)
        if j in inv and j + 1 < depth:
            do_round(label, True)
            label += 1
    if terminal_check:
        do_round(depth, False)          # repo: no reset after the terminal round
    segments.append(list(pending))
    pending.clear()

    rq = (0, 3) if readout == "data2" else DATA
    for q in rq:
        em.ideal("H", (q,))
    data_rec = em.ideal_measure(rq)

    raw: dict = {}
    prev = None
    for lab in rounds:
        recs = round_recs[lab]
        if reset or prev is None:
            raw[lab] = [frozenset([recs[s]]) for s in range(4)]
        else:
            raw[lab] = [frozenset([recs[s], round_recs[prev][s]])
                        for s in range(4)]
        prev = lab

    dets: list[tuple[set, int]] = []
    back: dict = {}
    for pos, lab in enumerate(rounds):
        if detectors == "diff" and pos > 0:
            subsets = tuple(pull_back_subset(segments[pos], s)
                            for s in range(4))
        else:
            subsets = tuple(() for _ in range(4))
        back[lab] = subsets
        for s in range(4):
            recs = set(raw[lab][s])
            for t in subsets[s]:
                recs ^= raw[rounds[pos - 1]][t]
            dets.append((recs, pos))
    n_round = len(dets)
    if readout == "data5":
        for s in X_STABS:
            recs = {data_rec[q] for q in support(s)}
            if detectors == "diff" and rounds:
                for t in pull_back_subset(segments[-1], s):
                    recs ^= raw[rounds[-1]][t]
            dets.append((recs, len(rounds)))
    total = em.nmeas
    for recs, _ in dets:
        em.c.append("DETECTOR", [stim.target_rec(r - total)
                                 for r in sorted(recs)])
    em.c.append("OBSERVABLE_INCLUDE",
                [stim.target_rec(data_rec[q] - total) for q in (0, 3)], 0)
    return MyCircuit(em.c, depth, rounds, back, segments,
                     np.array([s for _, s in dets], dtype=np.int64), n_round,
                     round_recs, data_rec, reset, dict(em.tally))


def unif_rounds(k: int, depth: int) -> list[int]:
    """Forward rounds of unif=k (own implementation; terminal excluded)."""
    return [j * k - 1 for j in range(1, depth // k + 1) if j * k - 1 < depth]


def const_rounds(k: int, depth: int) -> list[int]:
    if k > depth:
        return []
    interval = depth // k
    return sorted({j * interval - 1 for j in range(1, k)})


# Variant name -> builder kwargs as a function of depth.
SCHED_VARIANTS = {
    "serial": dict(schedule="serial", reset=True),
    "zx": dict(schedule="zx", reset=True),
    "parallel": dict(schedule="parallel", reset=True),
    "serial-nr": dict(schedule="serial", reset=False),
    "zx-nr": dict(schedule="zx", reset=False),
    "parallel-nr": dict(schedule="parallel", reset=False),
    "parallel_xz": dict(schedule="parallel_xz", reset=True),
    "parallel_xz-nr": dict(schedule="parallel_xz", reset=False),
}


def variant_kwargs(name: str, depth: int) -> dict:
    """Builder kwargs for a variant name such as ``parallel-nr``,
    ``parallel-nr@unif4``, ``serial@term``, ``serial@const2``,
    ``parallel@unif4+inv4``, ``const0``."""
    if name == "const0":
        return dict(schedule="serial", reset=True, forward=(),
                    terminal_check=False)
    sched, _, policy = name.partition("@")
    kw = dict(SCHED_VARIANTS[sched])
    if not policy or policy == "all":
        return kw
    fwd_part, _, inv_part = policy.partition("+")
    if fwd_part == "term":
        kw["forward"] = ()
    elif fwd_part.startswith("unif"):
        kw["forward"] = unif_rounds(int(fwd_part[4:]), depth)
    elif fwd_part.startswith("const"):
        kw["forward"] = const_rounds(int(fwd_part[5:]), depth)
    elif fwd_part != "all":
        raise ValueError(name)
    if inv_part:
        k = int(inv_part[3:])
        kw["inverse"] = [j for j in range(depth) if (j + 1) % k == 0
                         and j + 1 < depth]
    return kw


def build_variant(name, seq, depth, p, **extra) -> MyCircuit:
    kw = variant_kwargs(name, depth)
    kw.update(extra)
    return build(seq, depth, p, **kw)


# ---------------------------------------------------------------------------
# DEM census (own implementation)
# ---------------------------------------------------------------------------

def parse_dem(dem: stim.DetectorErrorModel):
    out = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        prob = inst.args_copy()[0]
        dets, flip = [], False
        for t in inst.targets_copy():
            if t.is_separator():
                raise AssertionError("unexpected decomposition separator")
            if t.is_relative_detector_id():
                dets.append(t.val)
            elif t.is_logical_observable_id():
                flip = not flip
        out.append((prob, tuple(sorted(dets)), flip))
    return out


def symptom_table(mechs, keep=None) -> dict:
    w: dict = defaultdict(lambda: [0.0, 0.0])
    for prob, dets, flip in mechs:
        if keep is not None:
            dets = tuple(d for d in dets if keep[d])
        w[dets][int(flip)] += prob
    return w


def census_numbers(w: dict, p: float) -> dict:
    e1 = r1 = a2 = 0.0
    u = w.get((), [0.0, 0.0])[1]
    for s, (p0, p1) in w.items():
        if not s:
            continue
        e1 += min(p0, p1)
        r1 += p0 + p1
        a2 += p0 * p1
    return {"E1": (e1 + u) / p, "R1": r1 / p, "A2": a2 / p ** 2, "U": u / p}


def census_circuit(mc: MyCircuit, p: float) -> dict:
    mechs = parse_dem(mc.circuit.detector_error_model())
    n = mc.circuit.num_detectors
    out = {f"{k}_5": v for k, v in census_numbers(symptom_table(mechs), p)
           .items()}
    keep = np.arange(n) < mc.n_round_dets
    out.update({f"{k}_2": v for k, v in census_numbers(
        symptom_table(mechs, keep), p).items()})
    return out


# ---------------------------------------------------------------------------
# Census jobs
# ---------------------------------------------------------------------------

SCAN_C = (0.0, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4, 1e-3, 1.5e-3, 2e-3, 3e-3,
          5e-3, 1e-2, 0.1, 1.0)
SCAN_VARIANTS = ("serial", "serial@unif4", "serial@unif11", "serial@term",
                 "parallel-nr", "parallel-nr@unif4", "parallel-nr@unif11",
                 "parallel-nr@term", "parallel", "zx")
INTERVAL_POLICIES = ("unif2", "unif3", "unif4", "unif6", "unif11", "term",
                     "const2", "const3", "const4")


def census_variant_names() -> list[str]:
    names = list(SCHED_VARIANTS) + ["const0"]
    for sched in ("serial", "parallel", "parallel-nr"):
        names += [f"{sched}@{pol}" for pol in INTERVAL_POLICIES]
        names += [f"{sched}@unif2+inv2", f"{sched}@unif4+inv4",
                  f"{sched}@unif11+inv11", f"{sched}@all+inv1"]
    return names


def _census_job(job):
    seqset, seq_index, seq, depths = job
    rows = []
    for depth in depths:
        for name in census_variant_names():
            mc = build_variant(name, seq, depth, P_SMALL, detectors="raw")
            row = {"seqset": seqset, "seq": seq_index, "variant": name,
                   "C": 1.0, "depth": depth, "rounds": mc.num_rounds}
            row.update(census_circuit(mc, P_SMALL))
            # data2 floor from an actual two-qubit-readout circuit
            if name != "const0" and "@" not in name:
                mc2 = build_variant(name, seq, depth, P_SMALL,
                                    detectors="raw", readout="data2")
                w2 = symptom_table(parse_dem(
                    mc2.circuit.detector_error_model()))
                row["U_data2circ"] = w2.get((), [0.0, 0.0])[1] / P_SMALL
            rows.append(row)
        # lead's builder (serial, every round) as a reference
        lc = Q.build_lrb_circuit(seq, depth, P_SMALL, terminal="data5")
        mechs = parse_dem(lc.circuit.detector_error_model())
        row = {"seqset": seqset, "seq": seq_index, "variant": "lead-serial",
               "C": 1.0, "depth": depth, "rounds": lc.num_rounds}
        n_round = 4 * lc.num_rounds
        keep = np.arange(lc.circuit.num_detectors) < n_round
        row.update({f"{k}_5": v for k, v in census_numbers(
            symptom_table(mechs), P_SMALL).items()})
        row.update({f"{k}_2": v for k, v in census_numbers(
            symptom_table(mechs, keep), P_SMALL).items()})
        rows.append(row)
        # check-noise scan (check rounds x C, logical gates fixed)
        for name in SCAN_VARIANTS:
            for cval in SCAN_C:
                if cval == 1.0:
                    continue
                mc = build_variant(name, seq, depth, P_SMALL,
                                   detectors="raw", check_scale=cval)
                row = {"seqset": seqset, "seq": seq_index,
                       "variant": f"scan:{name}", "C": cval, "depth": depth,
                       "rounds": mc.num_rounds}
                row.update(census_circuit(mc, P_SMALL))
                rows.append(row)
    return rows


def _monotone_job(job):
    """Random nested chains of check rounds at depth 22 (claim 9).

    Start from const0, add the terminal round, then add forward and inverse
    rounds one at a time in random order; record E1 at every step.
    """
    seqset, seq_index, seq, sched, chain_seed, depth = job
    rng = np.random.default_rng(chain_seed)
    candidates = [("f", j) for j in range(depth)] + \
                 [("i", j) for j in range(depth - 1)]
    order = [candidates[i] for i in rng.permutation(len(candidates))]
    kw = dict(SCHED_VARIANTS[sched])
    e1 = []
    mc = build(seq, depth, P_SMALL, schedule="serial", reset=True, forward=(),
               terminal_check=False, detectors="raw")
    e1.append(census_circuit(mc, P_SMALL)["E1_5"])
    fwd, inv = [], []
    mc = build(seq, depth, P_SMALL, forward=fwd, inverse=inv,
               detectors="raw", **kw)
    e1.append(census_circuit(mc, P_SMALL)["E1_5"])
    for kind, j in order:
        (fwd if kind == "f" else inv).append(j)
        mc = build(seq, depth, P_SMALL, forward=fwd, inverse=inv,
                   detectors="raw", **kw)
        e1.append(census_circuit(mc, P_SMALL)["E1_5"])
    steps = np.diff(e1)
    return {"seqset": seqset, "seq": seq_index, "schedule": sched,
            "chain_seed": chain_seed, "depth": depth,
            "order": [f"{k}{j}" for k, j in order], "E1": e1,
            "min_step": float(steps.min()),
            "num_negative_steps": int((steps < 0).sum())}


def _equal_count_job(job):
    """Inverse-half placement at an exactly equal number of rounds."""
    seqset, seq_index, seq = job
    depth = 22
    out = []
    for sched in ("serial", "parallel", "parallel-nr"):
        kw = dict(SCHED_VARIANTS[sched])
        fwd_only = list(range(1, 21, 2))                  # 10 forward rounds
        split_f = [3, 7, 11, 15, 19]                       # 5 forward
        split_i = [3, 7, 11, 15, 19]                       # 5 inverse
        a = census_circuit(build(seq, depth, P_SMALL, forward=fwd_only,
                                 detectors="raw", **kw), P_SMALL)
        b = census_circuit(build(seq, depth, P_SMALL, forward=split_f,
                                 inverse=split_i, detectors="raw", **kw),
                           P_SMALL)
        out.append({"seqset": seqset, "seq": seq_index, "schedule": sched,
                    "E1_fwd10": a["E1_5"], "E1_fwd5_inv5": b["E1_5"],
                    "R1_fwd10": a["R1_5"], "R1_fwd5_inv5": b["R1_5"],
                    "A2_fwd10": a["A2_5"], "A2_fwd5_inv5": b["A2_5"]})
    return out


def idle_budget() -> dict:
    """Per forward check round: channel mass by category (units of p) and the
    number of DEPOLARIZE1(2p) layers on each data qubit (claim 2)."""
    p = 1e-3
    out = {}
    for sched in SCHEDULES:
        for reset in (True, False):
            em = Emitter(p)
            check_round(em, sched, reset)
            layers = {q: 0 for q in DATA}
            for inst in em.c:
                if inst.name == "DEPOLARIZE1" and abs(
                        inst.gate_args_copy()[0] - 2 * p) < 1e-15:
                    for t in inst.targets_copy():
                        if t.value in DATA:
                            layers[t.value] += 1
            out[f"{sched}{'' if reset else '-nr'}"] = {
                "data_idle_layers": sorted(set(layers.values())),
                "mass_over_p": {k: round(v / p, 6)
                                for k, v in sorted(em.tally.items())}}
    # whole-circuit consistency: m forward rounds + terminal (no reset)
    seq = E.clifford_sequences(1, 22, NEW_SEED)[0]
    for sched, kw in SCHED_VARIANTS.items():
        if sched.startswith("parallel_xz"):
            continue
        mc = build(seq, 6, p, detectors="raw", **kw)
        count = {q: 0 for q in DATA}
        for inst in mc.circuit:
            if inst.name == "DEPOLARIZE1" and abs(
                    inst.gate_args_copy()[0] - 2 * p) < 1e-15:
                for t in inst.targets_copy():
                    if t.value in DATA:
                        count[t.value] += 1
        out[sched]["depth6_total_layers_per_data_qubit"] = sorted(
            set(count.values()))
    return out


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def slopes(table: np.ndarray) -> np.ndarray:
    """Least-squares slope over DEPTHS for each row of (n_seq, n_depth)."""
    return np.polyfit(np.asarray(DEPTHS, float), table.T, 1)[0]


def ms(values) -> dict:
    v = np.asarray(values, float)
    sem = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float(
        "nan")
    return {"mean": float(v.mean()), "sem": sem, "n": int(len(v))}


def table_of(rows, seqset, variant, key, n_seq, C_val=1.0):
    t = np.full((n_seq, len(DEPTHS)), np.nan)
    for r in rows:
        if (r["seqset"] == seqset and r["variant"] == variant
                and abs(r["C"] - C_val) < 1e-15):
            t[r["seq"], DEPTHS.index(r["depth"])] = r[key]
    return t


def load_census_files(out: Path):
    rows = []
    with open(out / "census_rows.csv") as fh:
        for r in csv.DictReader(fh):
            row = {}
            for k, v in r.items():
                if v == "":
                    continue
                if k in ("seqset", "variant"):
                    row[k] = v
                elif k in ("seq", "depth", "rounds"):
                    row[k] = int(v)
                else:
                    row[k] = float(v)
            rows.append(row)
    mono = json.loads((out / "monotone_chains.json").read_text())
    eq = json.loads((out / "equal_count_inverse.json").read_text())
    n_seq = {}
    for r in rows:
        n_seq[r["seqset"]] = max(n_seq.get(r["seqset"], 0), r["seq"] + 1)
    return rows, mono, eq, n_seq


def cmd_census_summary(args) -> int:
    out = Path(args.out)
    rows, mono, eq, n_seq = load_census_files(out)
    summary = summarize_census(rows, mono, eq, n_seq)
    summary["idle_budget"] = idle_budget()
    (out / "census_summary.json").write_text(json.dumps(summary, indent=1))
    print_census(summary)
    return 0


def cmd_census(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = []
    seqsets = {"new": E.clifford_sequences(args.num_new, 22, NEW_SEED),
               "theirs": E.clifford_sequences(args.num_theirs, 22,
                                              THEIR_SEED)}
    for name, seqs in seqsets.items():
        for i, s in enumerate(seqs):
            for depth in sorted(DEPTHS, reverse=True):
                jobs.append((name, i, s, [depth]))
    jobs.sort(key=lambda j: -j[3][0])      # longest (deepest) first
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(_census_job, jobs):
            rows.extend(part)
        print(f"census rows: {len(rows)} in {time.time() - t0:.0f}s",
              flush=True)
        mono_jobs = []
        for name, seqs in seqsets.items():
            if name != "new":
                continue
            for i, s in enumerate(seqs):
                for sched in ("serial", "parallel-nr", "parallel"):
                    for c in range(args.chains):
                        mono_jobs.append((name, i, s, sched,
                                          1000 * i + c, 22))
        mono = list(pool.map(_monotone_job, mono_jobs))
        print(f"monotone chains: {len(mono)} in {time.time() - t0:.0f}s",
              flush=True)
        eq = []
        for part in pool.map(_equal_count_job,
                             [(n, i, s) for n, seqs in seqsets.items()
                              for i, s in enumerate(seqs)]):
            eq.extend(part)
    keys = sorted({k for r in rows for k in r})
    with open(out / "census_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / "monotone_chains.json").write_text(json.dumps(mono, indent=1))
    (out / "equal_count_inverse.json").write_text(json.dumps(eq, indent=1))
    summary = summarize_census(rows, mono, eq,
                               {k: len(v) for k, v in seqsets.items()})
    summary["idle_budget"] = idle_budget()
    (out / "census_summary.json").write_text(json.dumps(summary, indent=1))
    print_census(summary)
    return 0


def summarize_census(rows, mono, eq, n_seq: dict) -> dict:
    summ: dict = {"p_small": P_SMALL, "depths": list(DEPTHS),
                  "n_seq": n_seq, "new_seed": NEW_SEED,
                  "their_seed": THEIR_SEED, "seqsets": {}}
    for seqset, n in n_seq.items():
        if n == 0:
            continue
        S: dict = {"slopes": {}, "paired": {}, "fractions": {},
                   "data2_floor": {}, "lead_match": {}, "scan": {}}
        variants = sorted({r["variant"] for r in rows
                           if r["seqset"] == seqset
                           and not r["variant"].startswith("scan:")})
        sl = {}
        for v in variants:
            entry = {}
            for key in ("E1_5", "E1_2", "R1_5", "A2_5"):
                t = table_of(rows, seqset, v, key, n)
                s = slopes(t)
                sl[(v, key)] = s
                entry[key] = ms(s)
                entry[key]["at_depth"] = {str(d): float(t[:, i].mean())
                                          for i, d in enumerate(DEPTHS)}
            S["slopes"][v] = entry
            if v not in ("const0", "lead-serial") and "@" not in v:
                t = table_of(rows, seqset, v, "U_data2circ", n)
                tm = table_of(rows, seqset, v, "U_2", n)
                S["data2_floor"][v] = {
                    "min": float(np.nanmin(t)), "max": float(np.nanmax(t)),
                    "mask_method_max_abs_diff": float(np.nanmax(np.abs(
                        t - tm)))}
        # lead's serial builder vs mine (same quantity, every seq and depth)
        for key in ("E1_5", "E1_2", "R1_5", "A2_5"):
            a = table_of(rows, seqset, "serial", key, n)
            b = table_of(rows, seqset, "lead-serial", key, n)
            S["lead_match"][key] = float(np.nanmax(np.abs(a - b)
                                                   / np.maximum(1, b)))
        e = {v: sl[(v, "E1_5")] for v in variants}
        for v in ("zx", "parallel", "parallel-nr", "serial-nr", "zx-nr",
                  "parallel_xz", "parallel_xz-nr", "const0"):
            S["paired"][f"{v} - serial"] = ms(e[v] - e["serial"])
        pen = e["serial"] - e["const0"]
        for v in ("zx", "parallel", "parallel-nr", "parallel_xz",
                  "parallel_xz-nr", "serial-nr", "zx-nr"):
            removed = e["serial"] - e[v]
            S["fractions"][v] = {
                "ratio_of_means": float(removed.mean() / pen.mean()),
                "per_sequence": ms(removed / pen)}
        # ordering by number of executed checks (claim 9)
        # scan: E1 slope vs C and crossover with const0
        c0 = e["const0"]
        for v in SCAN_VARIANTS:
            pts = []
            for cval in SCAN_C:
                name = v if cval == 1.0 else f"scan:{v}"
                t = table_of(rows, seqset, name, "E1_5", n, cval)
                if np.isnan(t).any():
                    continue
                s = slopes(t)
                pts.append((cval, ms(s), ms(s - c0)))
            cs = np.array([q[0] for q in pts])
            d = np.array([q[2]["mean"] for q in pts])
            cross = None
            for i in range(len(cs) - 1):
                if d[i] <= 0 < d[i + 1]:
                    cross = float(cs[i] + (cs[i + 1] - cs[i]) * (-d[i])
                                  / (d[i + 1] - d[i]))
                    break
            # linear interpolation between C=0 and C=0.01 only (implementer)
            d0 = d[cs == 0.0][0]
            d01 = d[np.isclose(cs, 0.01)][0]
            coarse = float(0.01 * (-d0) / (d01 - d0)) if d01 > d0 else None
            S["scan"][v] = {
                "points": [{"C": q[0], "E1": q[1], "dE1_vs_const0": q[2]}
                           for q in pts],
                "crossover_C": cross,
                "crossover_C_from_C0_and_C001_only": coarse,
                "quieter_factor": (1 / cross) if cross else None}
        summ["seqsets"][seqset] = S
    # monotone chains
    if mono:
        summ["monotone"] = {
            "num_chains": len(mono),
            "num_steps": int(sum(len(m["E1"]) - 1 for m in mono)),
            "num_negative_steps": int(sum(m["num_negative_steps"]
                                          for m in mono)),
            "min_step_over_p": float(min(m["min_step"] for m in mono)),
            "by_schedule": {
                sched: {"min_step": float(min(m["min_step"] for m in mono
                                              if m["schedule"] == sched)),
                        "negatives": int(sum(m["num_negative_steps"]
                                             for m in mono
                                             if m["schedule"] == sched))}
                for sched in sorted({m["schedule"] for m in mono})}}
        # which additions were cheapest
        cheapest = sorted(
            ((float(np.diff(m["E1"])[k]), m["schedule"], m["seq"],
              (["const0->term"] + ["term"] + m["order"])[k])
             for m in mono for k in range(len(m["E1"]) - 1)))[:8]
        summ["monotone"]["cheapest_additions"] = cheapest
    if eq:
        res = {}
        for seqset in sorted({r["seqset"] for r in eq}):
            for sched in ("serial", "parallel", "parallel-nr"):
                sel = [r for r in eq if r["seqset"] == seqset
                       and r["schedule"] == sched]
                if not sel:
                    continue
                res[f"{seqset}:{sched}"] = {
                    "E1_fwd10": ms([r["E1_fwd10"] for r in sel]),
                    "E1_fwd5_inv5": ms([r["E1_fwd5_inv5"] for r in sel]),
                    "dE1(inv - fwd)": ms([r["E1_fwd5_inv5"] - r["E1_fwd10"]
                                          for r in sel]),
                    "dR1(inv - fwd)": ms([r["R1_fwd5_inv5"] - r["R1_fwd10"]
                                          for r in sel]),
                    "dA2(inv - fwd)": ms([r["A2_fwd5_inv5"] - r["A2_fwd10"]
                                          for r in sel])}
        summ["equal_count_inverse_depth22"] = res
    return summ


def _f(x, nd=2):
    return f"{x['mean']:.{nd}f}({x['sem']:.{nd}f})"


def print_census(summary: dict) -> None:
    for seqset, S in summary["seqsets"].items():
        print(f"\n=== seqset {seqset} (n={summary['n_seq'][seqset]}) ===")
        print("lead_match (max rel diff serial vs lead):", S["lead_match"])
        print(f"{'variant':28s} {'E1_5/step':>12s} {'E1_2/step':>12s} "
              f"{'R1/step':>12s} {'A2/step':>12s}")
        for v, e in S["slopes"].items():
            print(f"{v:28s} {_f(e['E1_5']):>12s} {_f(e['E1_2']):>12s} "
                  f"{_f(e['R1_5']):>12s} {_f(e['A2_5']):>12s}")
        print("paired:", {k: _f(v, 3) for k, v in S["paired"].items()})
        print("fractions:", {k: round(v["ratio_of_means"], 4)
                             for k, v in S["fractions"].items()})
        print("data2 floor:", {k: (round(v["min"], 4), round(v["max"], 4))
                               for k, v in S["data2_floor"].items()
                               if "@" not in k})
        for v, sc in S["scan"].items():
            print(f"scan {v}: C*={sc['crossover_C']}, "
                  f"coarse={sc['crossover_C_from_C0_and_C001_only']}, "
                  f"dE1(C=0)={_f(sc['points'][0]['dE1_vs_const0'], 4)}")
    if "monotone" in summary:
        print("\nmonotone:", json.dumps(summary["monotone"], indent=0)[:1500])
    if "equal_count_inverse_depth22" in summary:
        for k, v in summary["equal_count_inverse_depth22"].items():
            print(k, {kk: _f(vv, 3) for kk, vv in v.items()})


# ---------------------------------------------------------------------------
# Exact LRB-D (lead's evaluator) for claims 11, 12 and 13
# ---------------------------------------------------------------------------

def exact_point(mc, used_slots=None, final_check=True):
    """(acceptance, P(accept and flip)) via the lead's exact evaluators.

    used_slots=None: every round slot used (cross-checked against
    exact_all_rounds)."""
    dem = mc.circuit.detector_error_model()
    slots = (mc.slots if isinstance(mc, MyCircuit)
             else M.lrb_detector_slots(mc))
    dec = M.TransferMatrixMLD(dem, slots)
    ev = X.RawPolicyEvaluator(mc, dec)
    if used_slots is None:
        a, af = ev.evaluate(range(mc.num_rounds), final_check)
        a2, af2 = X.exact_all_rounds(dec, mc.num_rounds, final_check)
        assert abs(a - a2) < 1e-12 and abs(af - af2) < 1e-12, (a, a2, af, af2)
        return a, af
    return ev.evaluate(used_slots, final_check)


# (label, builder variant or "lead", forward policy for the circuit, policy
#  used for acceptance on that circuit, final_check)
EXACT_CASES = [
    # claim 11: executed circuits, final data check (all executed rounds used)
    ("c11:parallel-nr@all", "parallel-nr", None, True),
    ("c11:parallel-nr@unif2", "parallel-nr@unif2", None, True),
    ("c11:parallel-nr@unif3", "parallel-nr@unif3", None, True),
    ("c11:parallel-nr@unif4", "parallel-nr@unif4", None, True),
    ("c11:parallel-nr@unif6", "parallel-nr@unif6", None, True),
    ("c11:parallel-nr@unif11", "parallel-nr@unif11", None, True),
    ("c11:parallel-nr@const2", "parallel-nr@const2", None, True),
    ("c11:parallel-nr@const3", "parallel-nr@const3", None, True),
    ("c11:parallel-nr@const4", "parallel-nr@const4", None, True),
    ("c11:parallel-nr@term", "parallel-nr@term", None, True),
    ("c11:parallel@all", "parallel", None, True),
    ("c11:parallel@unif4", "parallel@unif4", None, True),
    ("c11:parallel@unif11", "parallel@unif11", None, True),
    ("c11:serial@all", "serial", None, True),
    ("c11:serial@unif4", "serial@unif4", None, True),
    ("c11:serial@unif11", "serial@unif11", None, True),
    ("c11:const0", "const0", None, True),
    # claim 12: repo protocol (every round run, policy selects) vs executed,
    # data2 semantics (no final data check)
    ("c12:full-serial:const1", "lead", ("const", 1), False),
    ("c12:full-serial:unif4", "lead", ("unif", 4), False),
    ("c12:full-serial:unif11", "lead", ("unif", 11), False),
    ("c12:exec-serial:term", "lead@term", None, False),
    ("c12:exec-serial:unif4", "lead@unif4", None, False),
    ("c12:exec-serial:unif11", "lead@unif11", None, False),
    ("c12:full-parallel-nr:const1", "parallel-nr", ("const", 1), False),
    ("c12:full-parallel-nr:unif4", "parallel-nr", ("unif", 4), False),
    ("c12:exec-parallel-nr:term", "parallel-nr@term", None, False),
    ("c12:exec-parallel-nr:unif4", "parallel-nr@unif4", None, False),
    # same, with my builder for the serial rows (cross-check of the lead's)
    ("c12:full-serial-mine:const1", "serial", ("const", 1), False),
    ("c12:exec-serial-mine:term", "serial@term", None, False),
]


def _make_exact_circuit(spec: str, seq, depth: int, p: float):
    if spec.startswith("lead"):
        _, _, pol = spec.partition("@")
        if not pol:
            fwd = None
        elif pol == "term":
            fwd = ()
        else:
            fwd = unif_rounds(int(pol[4:]), depth)
        return Q.build_lrb_circuit(seq, depth, p, terminal="data5",
                                   forward_check_rounds=fwd)
    return build_variant(spec, seq, depth, p, detectors="diff")


def _exact_job(job):
    seqset, seq_index, seq, p, depths, cases = job
    rows = []
    for depth in depths:
        for label, spec, policy, final in cases:
            lc = _make_exact_circuit(spec, seq, depth, p)
            if policy is None:
                a, af = exact_point(lc, None, final)
            else:
                used = B.used_rounds(policy[0], policy[1], depth)
                mine = sorted(set(([depth] + (unif_rounds(policy[1], depth)
                                              if policy[0] == "unif" else
                                              const_rounds(policy[1],
                                                           depth)))))
                assert used == mine, (policy, depth, used, mine)
                slots = [lc.rounds.index(r) for r in used]
                a, af = exact_point(lc, slots, final)
            rows.append({"seqset": seqset, "seq": seq_index, "p": p,
                         "case": label, "depth": depth, "A": a, "AF": af,
                         "rounds": lc.num_rounds})
    return rows


def cmd_exact(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seqsets = {"theirs": E.clifford_sequences(args.num_theirs, 22,
                                              THEIR_SEED),
               "new": E.clifford_sequences(args.num_new, 22, NEW_SEED)}
    probs = [float(x) for x in args.probs.split(",")]
    jobs = []
    for p in probs:
        cases = EXACT_CASES if p == 5e-3 else [
            c for c in EXACT_CASES if c[0].startswith("c11:parallel")
            or c[0].startswith("c11:const0")]
        for name, seqs in seqsets.items():
            for i, s in enumerate(seqs):
                for depth in DEPTHS:
                    jobs.append((name, i, s, p, [depth], cases))
    jobs.sort(key=lambda j: -j[4][0])
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(_exact_job, jobs):
            rows.extend(part)
        inv_rows = []
        inv_jobs = [(name, i, s) for name, seqs in seqsets.items()
                    for i, s in enumerate(seqs)]
        for part in pool.map(_exact_inverse_job, inv_jobs):
            inv_rows.extend(part)
    print(f"exact rows: {len(rows)} in {time.time() - t0:.0f}s", flush=True)
    keys = list(rows[0].keys())
    with open(out / "exact_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    (out / "exact_inverse_rows.json").write_text(json.dumps(inv_rows,
                                                            indent=1))
    summary = summarize_exact(rows, inv_rows, seqsets)
    (out / "exact_summary.json").write_text(json.dumps(summary, indent=1))
    print_exact(summary)
    return 0


def _exact_inverse_job(job):
    seqset, seq_index, seq = job
    rows = []
    depth = 22
    for p in (2e-3, 5e-3):
        for sched in ("parallel", "parallel-nr"):
            kw = dict(SCHED_VARIANTS[sched])
            cases = {
                "fwd unif2 (11 rounds)": dict(forward=unif_rounds(2, depth)),
                "fwd10 (rounds 1,3,..,19)": dict(
                    forward=list(range(1, 21, 2))),
                "fwd5+inv5 (unif4 + inverse every 4)": dict(
                    forward=unif_rounds(4, depth),
                    inverse=[3, 7, 11, 15, 19]),
                "fwd unif4 (5 rounds)": dict(forward=unif_rounds(4, depth)),
                "fwd unif6 (3 rounds)": dict(forward=unif_rounds(6, depth)),
                "fwd unif11 (2 rounds)": dict(
                    forward=unif_rounds(11, depth)),
                "fwd unif6 + inv every 6 (6 rounds)": dict(
                    forward=unif_rounds(6, depth), inverse=[5, 11, 17]),
                "fwd unif8 + inv every 8 (4 rounds)": dict(
                    forward=unif_rounds(8, depth), inverse=[7, 15]),
                "fwd unif11 + inv every 11 (3 rounds)": dict(
                    forward=unif_rounds(11, depth), inverse=[10]),
                "fwd {10} + inv {10} (2 rounds)": dict(
                    forward=[10], inverse=[10]),
            }
            for label, ckw in cases.items():
                mc = build(seq, depth, p, detectors="diff", **kw, **ckw)
                a, af = exact_point(mc, None, True)
                rows.append({"seqset": seqset, "seq": seq_index, "p": p,
                             "schedule": sched, "case": label, "A": a,
                             "AF": af, "rounds": mc.num_rounds})
    return rows


def summarize_exact(rows, inv_rows, seqsets) -> dict:
    out: dict = {"depths": list(DEPTHS), "points": {}, "fits": {},
                 "inverse": {}}
    groups = defaultdict(list)
    for r in rows:
        groups[(r["seqset"], r["p"], r["case"])].append(r)
    for (seqset, p, case), sel in sorted(groups.items()):
        n = len(seqsets[seqset])
        A = np.full((n, len(DEPTHS)), np.nan)
        F = np.full((n, len(DEPTHS)), np.nan)
        for r in sel:
            i, j = r["seq"], DEPTHS.index(r["depth"])
            A[i, j] = r["A"]
            F[i, j] = abs(1 - 2 * r["AF"] / r["A"]) if r["A"] > 0 else np.nan
        key = f"{seqset}|{p:g}|{case}"
        entry = {
            "A_mean": np.nanmean(A, axis=0).tolist(),
            "F_mean": np.nanmean(F, axis=0).tolist(),
            "F_std": np.nanstd(F, axis=0).tolist(),
            "infid22": ms(1 - F[:, -1]),
            "A22": ms(A[:, -1]),
        }
        fit = FT.fit_fixed_offset(DEPTHS, entry["F_mean"], entry["F_std"])
        entry["fit_f_repo"] = None if fit is None else fit["f"]
        # same fit with a floor on the per-depth std (avoids pinning F(0))
        std = np.maximum(np.asarray(entry["F_std"]), 1e-4)
        fit2 = FT.fit_fixed_offset(DEPTHS, entry["F_mean"], std)
        entry["fit_f_floor1e-4"] = None if fit2 is None else fit2["f"]
        out["points"][key] = entry
    groups = defaultdict(list)
    for r in inv_rows:
        groups[(r["seqset"], r["p"], r["schedule"], r["case"])].append(r)
    for (seqset, p, sched, case), sel in sorted(groups.items()):
        a = np.array([r["A"] for r in sel])
        f = np.array([abs(1 - 2 * r["AF"] / r["A"]) for r in sel])
        out["inverse"][f"{seqset}|{p:g}|{sched}|{case}"] = {
            "infid22": ms(1 - f), "A22": ms(a),
            "rounds": sel[0]["rounds"]}
    return out


def print_exact(summary: dict) -> None:
    print(f"{'seqset|p|case':52s} {'100(1-F22)':>14s} {'A22':>16s} "
          f"{'f_repo':>9s} {'f_floor':>9s}")
    for key, e in summary["points"].items():
        print(f"{key:52s} {100 * e['infid22']['mean']:8.4f}"
              f"({100 * e['infid22']['sem']:.4f}) "
              f"{e['A22']['mean']:10.5g}({e['A22']['sem']:.2g}) "
              f"{e['fit_f_repo'] or float('nan'):9.5f} "
              f"{e['fit_f_floor1e-4'] or float('nan'):9.5f}")
    for key, e in summary["inverse"].items():
        print(f"{key:70s} 100(1-F22)={100 * e['infid22']['mean']:.4f}"
              f"({100 * e['infid22']['sem']:.4f}) A22={e['A22']['mean']:.4f}"
              f" rounds={e['rounds']}")


# ---------------------------------------------------------------------------
# Finite-p gain of NOISELESS checks, paired on identical shots (claim 10)
# ---------------------------------------------------------------------------

def _paired_job(job):
    """Perfect check rounds (check noise x 0) add no noise, so the circuit's
    noise equals the const0 circuit's. Decode the same shots with every
    detector (exact MLD) and with only the raw final X parities (exact
    const0 MLD); the error difference is the exact finite-p value of the
    information the noiseless checks carry."""
    seqset, seq_index, seq, p, depth, shots, seed = job
    c0 = build(seq, depth, p, schedule="serial", reset=True, forward=(),
               terminal_check=False, detectors="raw")
    dec0 = M.TransferMatrixMLD(c0.circuit.detector_error_model(), c0.slots)
    rows = []
    for label, fwd in (("perfect-every-round", None),
                       ("perfect-terminal-only", ())):
        mc = build(seq, depth, p, schedule="serial", reset=True,
                   forward=fwd, detectors="diff", check_scale=0.0)
        dec = M.TransferMatrixMLD(mc.circuit.detector_error_model(),
                                  mc.slots)
        dets, obs = mc.circuit.compile_detector_sampler(
            seed=seed + (0 if fwd is None else 1)).sample(
                shots, separate_observables=True)
        flip = obs[:, 0]
        err_full = (dec.log_likelihood_ratio(dets) < 0) != flip
        raws = []
        for k, lab in enumerate(mc.rounds):
            cur = dets[:, 4 * k:4 * k + 4].copy()
            if k > 0:
                for s in range(4):
                    for t in mc.back[lab][s]:
                        cur[:, s] ^= raws[k - 1][:, t]
            raws.append(cur)
        fin = dets[:, mc.n_round_dets:].copy()
        for row, s in enumerate(X_STABS):
            for t in pull_back_subset(mc.segments[-1], s):
                fin[:, row] ^= raws[-1][:, t]
        err_c0 = (dec0.log_likelihood_ratio(fin) < 0) != flip
        rows.append({"seqset": seqset, "seq": seq_index, "p": p,
                     "depth": depth, "case": label, "shots": shots,
                     "err_checks": int(err_full.sum()),
                     "err_const0": int(err_c0.sum()),
                     "disagree": int((err_full != err_c0).sum())})
    return rows


def cmd_paired(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    seqs = E.clifford_sequences(args.num_new, 22, NEW_SEED)
    probs = [float(x) for x in args.probs.split(",")]
    jobs = []
    for p in probs:
        for i, s in enumerate(seqs):
            for depth in DEPTHS:
                jobs.append(("new", i, s, p, depth, args.shots,
                             int(1e6 * p) * 100_000 + 1000 * i + depth))
    jobs.sort(key=lambda j: -j[4])
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(_paired_job, jobs):
            rows.extend(part)
    print(f"paired rows: {len(rows)} in {time.time() - t0:.0f}s", flush=True)
    (out / "paired_perfect_checks_rows.json").write_text(json.dumps(rows))
    n = len(seqs)
    rng = np.random.default_rng(5)
    boot_idx = rng.integers(0, n, size=(500, n))
    summary = {}
    for p in probs:
        for case in ("perfect-every-round", "perfect-terminal-only"):
            Fc = np.zeros((n, len(DEPTHS)))
            F0 = np.zeros((n, len(DEPTHS)))
            dE = np.zeros((n, len(DEPTHS)))
            for r in rows:
                if r["p"] != p or r["case"] != case:
                    continue
                j = DEPTHS.index(r["depth"])
                Fc[r["seq"], j] = 1 - 2 * r["err_checks"] / r["shots"]
                F0[r["seq"], j] = 1 - 2 * r["err_const0"] / r["shots"]
                dE[r["seq"], j] = (r["err_checks"] - r["err_const0"]) / r[
                    "shots"]

            def fit_r(F):
                fit = FT.fit_fixed_offset(DEPTHS, F.mean(axis=0),
                                          F.std(axis=0))
                return (1 - fit["f"]) / 2 / p

            d = fit_r(Fc) - fit_r(F0)
            boots = [fit_r(Fc[b]) - fit_r(F0[b]) for b in boot_idx]
            slope = slopes(dE) / p
            summary[f"{p:g}|{case}"] = {
                "r_checks_over_p": fit_r(Fc), "r_const0_over_p": fit_r(F0),
                "dr_over_p_fit": d, "dr_over_p_fit_boot_std": float(
                    np.std(boots, ddof=1)),
                "d_err_slope_over_p": ms(slope),
                "d_err_at_22": ms(dE[:, -1])}
    (out / "paired_perfect_checks_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


# ---------------------------------------------------------------------------
# Finite-p decoded rates with my circuits (claim 6, sampled penalty fractions)
# ---------------------------------------------------------------------------

SAMPLE_VARIANTS = ("serial", "parallel", "parallel-nr", "const0")


def _sample_job(job):
    seq_index, seq, p, depth, shots, seed = job
    rows = []
    for k, name in enumerate(SAMPLE_VARIANTS):
        mc = build_variant(name, seq, depth, p, detectors="diff")
        dec = M.TransferMatrixMLD(mc.circuit.detector_error_model(),
                                  mc.slots)
        dets, obs = mc.circuit.compile_detector_sampler(
            seed=seed + 7919 * k).sample(shots, separate_observables=True)
        err = int(((dec.log_likelihood_ratio(dets) < 0) != obs[:, 0]).sum())
        rows.append({"seq": seq_index, "p": p, "depth": depth,
                     "variant": name, "shots": shots, "errors": err})
    return rows


def cmd_sample(args) -> int:
    out = Path(args.out)
    seqs = E.clifford_sequences(args.num_new, 22, NEW_SEED)
    probs = [float(x) for x in args.probs.split(",")]
    jobs = [(i, s, p, d, args.shots, 900_000_000 + int(p * 1e7) * 1000
             + 37 * i + d) for p in probs for i, s in enumerate(seqs)
            for d in DEPTHS]
    jobs.sort(key=lambda j: -j[3])
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for part in pool.map(_sample_job, jobs):
            rows.extend(part)
    print(f"sample rows: {len(rows)} in {time.time() - t0:.0f}s", flush=True)
    (out / "sample_rows.json").write_text(json.dumps(rows))
    n = len(seqs)
    boot = np.random.default_rng(11).integers(0, n, size=(500, n))
    summary = {}
    for p in probs:
        F = {}
        for v in SAMPLE_VARIANTS:
            t = np.zeros((n, len(DEPTHS)))
            for r in rows:
                if r["p"] == p and r["variant"] == v:
                    t[r["seq"], DEPTHS.index(r["depth"])] = abs(
                        1 - 2 * r["errors"] / r["shots"])
            F[v] = t

        def rate(t):
            fit = FT.fit_fixed_offset(DEPTHS, t.mean(axis=0), t.std(axis=0))
            return (1 - fit["f"]) / 2 / p

        point = {v: rate(F[v]) for v in SAMPLE_VARIANTS}
        bs = {v: np.array([rate(F[v][b]) for b in boot])
              for v in SAMPLE_VARIANTS}
        entry = {"r_over_p": point,
                 "r_over_p_boot_std": {v: float(bs[v].std(ddof=1))
                                       for v in SAMPLE_VARIANTS}}
        for v in ("parallel", "parallel-nr"):
            frac = (bs["serial"] - bs[v]) / (bs["serial"] - bs["const0"])
            entry[f"penalty_fraction_{v}"] = {
                "point": (point["serial"] - point[v])
                / (point["serial"] - point["const0"]),
                "boot_std": float(frac.std(ddof=1))}
        summary[f"{p:g}"] = entry
    (out / "sample_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


# ---------------------------------------------------------------------------
# Self-test of the builder and evaluator
# ---------------------------------------------------------------------------

def dem_dict(dem) -> dict:
    w = symptom_table(parse_dem(dem))
    return {k: tuple(v) for k, v in w.items()}


def cmd_selftest(args) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report: dict = {}
    seqs = E.clifford_sequences(3, 22, NEW_SEED + 99)
    # 1. noiseless determinism and DEM buildability for every variant
    n_ok = 0
    for name in list(SCHED_VARIANTS) + ["const0", "parallel-nr@unif4+inv4",
                                        "serial-nr@unif3+inv2"]:
        for depth in (0, 1, 3, 6, 22):
            for readout in ("data2", "data5"):
                for det in ("raw", "diff"):
                    if name == "const0" and readout == "data2":
                        continue
                    mc = build_variant(name, seqs[0], depth, 0.0,
                                       detectors=det, readout=readout)
                    d, o = mc.circuit.compile_detector_sampler(
                        seed=1).sample(64, separate_observables=True)
                    assert not d.any() and not o.any(), (name, depth)
                    noisy = build_variant(name, seqs[0], depth, 0.01,
                                          detectors=det, readout=readout)
                    dem = noisy.circuit.detector_error_model()
                    if det == "diff":
                        M.TransferMatrixMLD(dem, noisy.slots)  # locality
                    n_ok += 1
    report["deterministic_and_buildable_cases"] = n_ok
    # 2. my serial (diff detectors) == the lead's DEM, symptom by symptom
    worst = 0.0
    n_cmp = 0
    for seq in seqs:
        for depth, fwd, inv in ((0, None, 0), (2, None, 0), (6, None, 0),
                                (22, None, 0), (10, [2, 5, 8], 0),
                                (6, [], 0), (6, None, 2), (22, [], 0)):
            ref = Q.build_lrb_circuit(seq, depth, 2e-3, terminal="data5",
                                      forward_check_rounds=fwd,
                                      inverse_check_every=inv)
            inverse = ([j for j in range(depth) if (j + 1) % inv == 0
                        and j + 1 < depth] if inv else ())
            mine = build(seq, depth, 2e-3, forward=fwd, inverse=inverse)
            a = dem_dict(ref.circuit.detector_error_model())
            b = dem_dict(mine.circuit.detector_error_model())
            assert a.keys() == b.keys(), (depth, fwd, inv)
            for k in a:
                worst = max(worst, abs(a[k][0] - b[k][0]),
                            abs(a[k][1] - b[k][1]))
            n_cmp += 1
    report["serial_dem_equals_lead"] = {"cases": n_cmp,
                                        "max_abs_prob_diff": worst}
    # 3. raw and diff detector bases give the same census
    worst = 0.0
    for name in SCHED_VARIANTS:
        for depth in (0, 3, 10):
            a = census_circuit(build_variant(name, seqs[1], depth, P_SMALL,
                                             detectors="raw"), P_SMALL)
            b = census_circuit(build_variant(name, seqs[1], depth, P_SMALL,
                                             detectors="diff"), P_SMALL)
            for k in a:
                worst = max(worst, abs(a[k] - b[k]) / max(1.0, abs(b[k])))
    report["raw_vs_diff_census_max_rel_diff"] = worst
    # 4. exact evaluator vs brute force on small no-reset / reset circuits
    worst = 0.0
    for name in ("parallel-nr", "serial-nr", "parallel", "zx-nr"):
        for depth in (1, 2):
            mc = build_variant(name, seqs[2], depth, 0.02, detectors="diff")
            dem = mc.circuit.detector_error_model()
            joint = M.brute_force_joint(dem)
            n = dem.num_detectors
            idx = np.arange(2 ** n)
            bits = ((idx[:, None] >> np.arange(n)) & 1).astype(bool)
            # raw syndromes from the difference detectors (own recursion)
            raw = []
            for k, lab in enumerate(mc.rounds):
                cur = bits[:, 4 * k:4 * k + 4].copy()
                if k > 0:
                    for s in range(4):
                        for t in mc.back[lab][s]:
                            cur[:, s] ^= raw[k - 1][:, t]
                raw.append(cur)
            final_raw = bits[:, mc.n_round_dets:].copy()
            for row, s in enumerate(X_STABS):
                for t in pull_back_subset(mc.segments[-1], s):
                    final_raw[:, row] ^= raw[-1][:, t]
            for used in ([len(mc.rounds) - 1], list(range(len(mc.rounds))),
                         [0]):
                for final in (False, True):
                    acc = np.ones(2 ** n, bool)
                    for k in used:
                        acc &= ~raw[k].any(axis=1)
                    if final:
                        acc &= ~final_raw.any(axis=1)
                    pa = joint[:2 ** n][acc].sum() + joint[2 ** n:][acc].sum()
                    paf = joint[2 ** n:][acc].sum()
                    a, af = exact_point(mc, used, final)
                    worst = max(worst, abs(pa - a), abs(paf - af))
    report["raw_policy_evaluator_vs_brute_force_max_abs_diff"] = worst
    # 5. exact vs Monte Carlo on measurement records (no detectors used)
    mc_rows = []
    for name, p in (("parallel-nr@unif4", 5e-3), ("parallel-nr", 5e-3),
                    ("serial", 5e-3)):
        depth = 10
        mc = build_variant(name, seqs[0], depth, p, detectors="diff")
        shots = 400_000
        meas = mc.circuit.compile_sampler(seed=7).sample(shots)
        ok = np.ones(shots, bool)
        prev = None
        for lab in mc.rounds:
            recs = mc.round_records[lab]
            for s in range(4):
                v = meas[:, recs[s]].copy()
                if not mc.reset and prev is not None:
                    v ^= meas[:, mc.round_records[prev][s]]
                ok &= ~v
            prev = lab
        par_f = np.zeros(shots, bool)
        for s in X_STABS:
            v = np.zeros(shots, bool)
            for q in support(s):
                v ^= meas[:, mc.data_records[q]]
            par_f |= v
        acc = ok & ~par_f
        flip = meas[:, mc.data_records[0]] ^ meas[:, mc.data_records[3]]
        a_mc, af_mc = acc.mean(), (acc & flip).mean()
        a, af = exact_point(mc, None, True)
        se_a = np.sqrt(a * (1 - a) / shots)
        se_af = np.sqrt(af * (1 - af) / shots)
        mc_rows.append({"variant": name, "p": p, "depth": depth,
                        "A_exact": a, "A_mc": float(a_mc),
                        "z_A": float((a_mc - a) / se_a),
                        "AF_exact": af, "AF_mc": float(af_mc),
                        "z_AF": float((af_mc - af) / se_af)})
    report["exact_vs_measurement_record_mc"] = mc_rows
    (out / "selftest.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))
    return 0


# ---------------------------------------------------------------------------
# Comparison with the implementer's variants.py (run last)
# ---------------------------------------------------------------------------

def cmd_compare(args) -> int:
    from lrb_qubit import variants as V   # only here, as instructed
    out = Path(args.out)
    report: dict = {}
    seqs = E.clifford_sequences(4, 22, NEW_SEED)
    pairs = [("serial", dict(schedule="serial", reset=True)),
             ("zx", dict(schedule="zx", reset=True)),
             ("parallel", dict(schedule="parallel", reset=True)),
             ("parallel-nr", dict(schedule="parallel", reset=False)),
             ("serial-nr", dict(schedule="serial", reset=False)),
             ("zx-nr", dict(schedule="zx", reset=False))]
    worst = {}
    for name, vkw in pairs:
        w = 0.0
        for seq in seqs:
            for depth in (0, 2, 6, 22):
                for fwd in (None, unif_rounds(4, depth)):
                    theirs = V.build_variant_circuit(
                        seq, depth, 2e-3, forward_check_rounds=fwd, **vkw)
                    mine = build(seq, depth, 2e-3, forward=fwd,
                                 detectors="diff", **SCHED_VARIANTS[name])
                    a = dem_dict(theirs.circuit.detector_error_model())
                    b = dem_dict(mine.circuit.detector_error_model())
                    if a.keys() != b.keys():
                        w = float("inf")
                        continue
                    for k in a:
                        w = max(w, abs(a[k][0] - b[k][0]),
                                abs(a[k][1] - b[k][1]))
                    assert theirs.rounds == mine.rounds
        worst[name] = w
    report["dem_max_abs_diff_vs_variants_py"] = worst
    # their first-order and exact-claim numbers vs mine
    their_fo = json.loads((ROOT / "studies/correction-decoders/qubit/"
                           "results/H/first_order_summary.json").read_text())
    my_fo = json.loads((out / "census_summary.json").read_text())
    comp = {}
    name_map = {"sched-serial": "serial", "sched-zx": "zx",
                "sched-parallel": "parallel",
                "sched-parallel-nr": "parallel-nr", "const0": "const0"}
    for theirs, mine in name_map.items():
        t = their_fo["variants"][theirs]["E1_5"]
        for seqset in ("theirs", "new"):
            m = my_fo["seqsets"][seqset]["slopes"][mine]["E1_5"]
            comp[f"{mine}|{seqset}"] = {
                "theirs": [t["slope_mean"], t["slope_sem"]],
                "mine": [m["mean"], m["sem"]],
                "z": (m["mean"] - t["slope_mean"]) / np.hypot(
                    m["sem"], t["slope_sem"]) if seqset == "new" else
                m["mean"] - t["slope_mean"]}
    report["first_order_E1_slopes"] = comp
    (out / "compare.json").write_text(json.dumps(report, indent=1,
                                                 default=float))
    print(json.dumps(report, indent=1, default=float))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("selftest", "census", "census-summary", "exact", "paired",
                 "sample", "compare"):
        sp = sub.add_parser(name)
        sp.add_argument("--out", default=str(OUT_DEFAULT))
        sp.add_argument("--workers", type=int, default=4)
        sp.add_argument("--num-new", type=int, default=10)
        sp.add_argument("--num-theirs", type=int, default=10)
        sp.add_argument("--chains", type=int, default=2)
        sp.add_argument("--probs", default="5e-3,2e-3")
        sp.add_argument("--shots", type=int, default=200_000)
    args = ap.parse_args()
    return {"selftest": cmd_selftest, "census": cmd_census,
            "census-summary": cmd_census_summary,
            "exact": cmd_exact, "paired": cmd_paired, "sample": cmd_sample,
            "compare": cmd_compare}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
