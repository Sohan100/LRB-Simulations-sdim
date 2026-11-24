"""Adversarial verification of work package I (erasure heralding, leakage, flags).

Independence: the erasure, first-order and flag checks never import
``lrb_qubit.erasure`` (except ``flagE1``, which needs the implementer's flagged
circuits as the object under test, and decodes them with Stim's DEM instead of
the implementer's probe enumeration). Signatures come from Stim's
``detecting_regions`` (sensitivity of every detector to a Pauli at a TICK),
which is a different code path from the implementer's ``FlipSimulator`` probe.
``lrb_qubit.leakage`` is used only in ``leaksim`` (new seeds of the
implementer's own simulator, to test a consequence of its claims) and its raw
result parts are read in ``leaktables``.

Subcommands (outputs in ``--out``, default ``results/I-verify``):

``capacity``   Code-capacity erasure correctability by binary symplectic
    algebra (own stabilizer/logical strings from CONTEXT.md).
``erasure``    Circuit level: for every 2q-gate and measurement-idle noise
    location of the lead's data5 circuit, all 16 (4) Paulis; a heralded
    erasure is correctable iff equal detector signatures imply equal
    observable flips. Also prints one hand-checkable location.
``census``     First-order E1 = sum_s min(P0, P1) from Stim's DEM (the lead's
    ``first_order.py`` approach) with SI1000 components scaled; gives the
    heralding reductions and the naive "2q-only" ratio.
``flags``      PauliString propagation through the X- and Z-check ancilla
    circuits with and without flag CNOTs; scan of all two-CNOT placements.
``flagE1``     E1 of the implementer's flagged circuits from Stim's DEM.
``leakfo``     Exact first-order effect of ONE leak event on LRB-D unif=1 +
    final check (u1f) and const=1, by GF(2) algebra over the uniformly random
    components the leakage model injects (partner Paulis, forced readouts).
``leaktables`` Recompute WP I leakage table entries from the raw parts.
``leaksim``    New-seed simulations: LRB-D u1f under leakage with many shots;
    const=1 control with extra Pauli noise instead of leakage.

Example::

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python studies/correction-decoders/qubit/scripts/verify_leak.py all
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np  # noqa: E402
import stim  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

DEFAULT_OUT = ROOT / "studies/correction-decoders/qubit/results/I-verify"
IRES = ROOT / "studies/correction-decoders/qubit/results/I"
SEQ_SEED = 20261028
DEPTHS_FO = [0, 2, 4, 6, 10, 22]
NOISE = {"DEPOLARIZE1", "DEPOLARIZE2", "X_ERROR", "Z_ERROR"}
TWO_Q = {"CX", "CZ", "SWAP"}

# Own copies of the code definitions (CONTEXT.md), not imported.
STABS = ("ZZ_Z_", "_ZZ_Z", "XXX__", "_X_XX")
LOG_X, LOG_Z = "X__X_", "Z_Z__"


def _dump(out: Path, name: str, obj) -> None:
    out.mkdir(parents=True, exist_ok=True)

    def clean(o):
        if isinstance(o, dict):
            return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)):
            return None if not np.isfinite(o) else float(o)
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, np.bool_):
            return bool(o)
        return o

    (out / name).write_text(json.dumps(clean(obj), indent=1))


# ---------------------------------------------------------------------------
# 1. Code capacity (binary symplectic algebra)
# ---------------------------------------------------------------------------


def _vec(s: str) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([c in "XY" for c in s], dtype=np.int64)
    z = np.array([c in "ZY" for c in s], dtype=np.int64)
    return x, z


def _sym(a, b) -> int:
    return int((a[0] @ b[1] + a[1] @ b[0]) % 2)


def _mul(a, b):
    return (a[0] ^ b[0], a[1] ^ b[1])


def _on(n: int, assignment: dict[int, str]) -> str:
    s = ["_"] * n
    for q, c in assignment.items():
        s[q] = c
    return "".join(s)


def cmd_capacity(args) -> dict:
    stabs = [_vec(s) for s in STABS]
    lx, lz = _vec(LOG_X), _vec(LOG_Z)
    # sanity: stabilizers commute, logicals commute with stabilizers, anticommute
    assert all(_sym(a, b) == 0 for a in stabs for b in stabs)
    assert all(_sym(s, lx) == 0 and _sym(s, lz) == 0 for s in stabs)
    assert _sym(lx, lz) == 1

    def syndrome(p):
        return tuple(_sym(p, s) for s in stabs)

    def logical_class(p):
        return (_sym(p, lz), _sym(p, lx))  # (flips X_L-eigen?, flips Z_L)

    def correctable(support):
        groups: dict = defaultdict(set)
        for letters in itertools.product("IXYZ", repeat=len(support)):
            p = _vec(_on(5, {q: c for q, c in zip(support, letters)
                             if c != "I"}))
            groups[syndrome(p)].add(logical_class(p))
        return all(len(v) == 1 for v in groups.values()), len(groups)

    single = {}
    for q in range(5):
        syn = {c: syndrome(_vec(_on(5, {q: c}))) for c in "XYZ"}
        distinct = len(set(syn.values())) == 3 and (0, 0, 0, 0) not in syn.values()
        single[q] = {"syndromes": {c: list(v) for c, v in syn.items()},
                     "distinct_nonzero": distinct,
                     "correctable": correctable((q,))[0]}
    pairs = {}
    for a, b in itertools.combinations(range(5), 2):
        ok, ngroups = correctable((a, b))
        pairs[f"{a}{b}"] = ok
    result = {"single": single, "pairs_correctable": pairs}
    print("code capacity, single-qubit erasures:")
    for q, r in single.items():
        print(f"  q{q}: {r['syndromes']}  distinct&nonzero={r['distinct_nonzero']}"
              f" correctable={r['correctable']}")
    print("  two-qubit erasure sets correctable:",
          {k: v for k, v in pairs.items()})
    _dump(Path(args.out), "capacity.json", result)
    return result


# ---------------------------------------------------------------------------
# Shared: TICK-annotated circuits and detecting-region sensitivities
# ---------------------------------------------------------------------------


class Sensitivity:
    """Per-tick Pauli sensitivity of every detector and of L0.

    A TICK is inserted after every noise instruction of ``circuit``; an error
    Pauli at that TICK flips target j iff it anticommutes with region j there.
    ``locs`` lists the noise instructions with their tick, channel, argument,
    targets and the next non-noise instruction name.
    """

    def __init__(self, circuit: stim.Circuit):
        self.circuit = circuit
        new = stim.Circuit()
        instrs = list(circuit)
        self.locs = []
        tick = 0
        for i, inst in enumerate(instrs):
            new.append(inst)
            if inst.name in NOISE:
                j = i + 1
                while j < len(instrs) and instrs[j].name in NOISE:
                    j += 1
                nxt = instrs[j].name if j < len(instrs) else None
                self.locs.append({
                    "index": i, "tick": tick, "name": inst.name,
                    "arg": inst.gate_args_copy()[0],
                    "targets": [t.value for t in inst.targets_copy()],
                    "next": nxt, "prev": instrs[i - 1].name if i else None})
                new.append("TICK")
                tick += 1
        self.annotated = new
        self.num_ticks = tick
        self.num_dets = circuit.num_detectors
        self.num_qubits = circuit.num_qubits
        regions = new.detecting_regions()
        n_t = self.num_dets + 1
        # sx[t, q, j]: X at (tick t, qubit q) flips target j; sz likewise for Z.
        self.sx = np.zeros((tick, self.num_qubits, n_t), dtype=bool)
        self.sz = np.zeros((tick, self.num_qubits, n_t), dtype=bool)
        for key, per_tick in regions.items():
            if key.is_logical_observable_id():
                if key.val != 0:
                    continue
                j = self.num_dets
            else:
                j = key.val
            for t, ps in per_tick.items():
                if t >= tick:
                    continue
                for q in range(min(len(ps), self.num_qubits)):
                    c = ps[q]  # 0 I, 1 X, 2 Y, 3 Z
                    if c in (2, 3):
                        self.sx[t, q, j] = True
                    if c in (1, 2):
                        self.sz[t, q, j] = True

    def component(self, tick: int, qubit: int, pauli: str) -> np.ndarray:
        return (self.sx if pauli == "X" else self.sz)[tick, qubit].copy()

    def pauli(self, tick: int, assignment: dict[int, int]) -> np.ndarray:
        """Signature (dets + L0) of a Pauli {qubit: 1 X, 2 Y, 3 Z}."""
        v = np.zeros(self.num_dets + 1, dtype=bool)
        for q, c in assignment.items():
            if c in (1, 2):
                v ^= self.sx[tick, q]
            if c in (2, 3):
                v ^= self.sz[tick, q]
        return v


def _sequences(n: int) -> list[list[str]]:
    return E.clifford_sequences(n, 22, SEQ_SEED)


# ---------------------------------------------------------------------------
# 2. Circuit-level single heralded erasures
# ---------------------------------------------------------------------------


def _erasure_locations(sens: Sensitivity, p: float):
    """Converted locations: every DEPOLARIZE2 (2q gates) and every qubit of a
    measurement-idle DEPOLARIZE1(2p) (next non-noise op is M)."""
    out = []
    for loc in sens.locs:
        if loc["name"] == "DEPOLARIZE2":
            t = loc["targets"]
            for a, b in zip(t[::2], t[1::2]):
                kind = ("cx_check" if max(a, b) >= 5 else "gate2_logical")
                out.append((kind, loc["tick"], (a, b)))
        elif (loc["name"] == "DEPOLARIZE1" and loc["next"] == "M"
              and abs(loc["arg"] - 2 * p) < 1e-12 * max(1.0, p)):
            for q in loc["targets"]:
                out.append(("meas_idle", loc["tick"], (q,)))
    return out


def _check_location(sens: Sensitivity, tick: int, qubits) -> tuple[bool, int]:
    """(ambiguous?, number of distinct detector signatures) over all Paulis."""
    groups: dict = defaultdict(set)
    for paulis in itertools.product(range(4), repeat=len(qubits)):
        v = sens.pauli(tick, dict(zip(qubits, paulis)))
        groups[v[:-1].tobytes()].add(bool(v[-1]))
    return any(len(s) > 1 for s in groups.values()), len(groups)


def cmd_erasure(args) -> dict:
    p = 1e-3
    seqs = _sequences(args.num_seq)
    counts = defaultdict(int)
    ambiguous = defaultdict(int)
    extra_counts = defaultdict(int)
    extra_amb = defaultdict(int)
    nsig = defaultdict(list)
    for si, seq in enumerate(seqs):
        for depth in DEPTHS_FO:
            lc = Q.build_lrb_circuit(seq, depth, p, terminal="data5")
            sens = Sensitivity(lc.circuit)
            for kind, tick, qubits in _erasure_locations(sens, p):
                amb, ng = _check_location(sens, tick, qubits)
                counts[kind] += 1
                ambiguous[kind] += int(amb)
                nsig[kind].append(ng)
            # Not claimed by WP I: the other channels, if they were heralded.
            for loc in sens.locs:
                if loc["name"] == "DEPOLARIZE1" and not (
                        loc["next"] == "M" and abs(loc["arg"] - 2 * p) < 1e-12):
                    kind = ("reset_idle" if loc["next"] == "R" else
                            "meas_depol" if loc["next"] == "M" else "gate1")
                    for q in loc["targets"]:
                        amb, _ = _check_location(sens, loc["tick"], (q,))
                        extra_counts[kind] += 1
                        extra_amb[kind] += int(amb)
        print(f"  seq {si}: cumulative locations {dict(counts)}", flush=True)
    total_2q = counts["cx_check"] + counts["gate2_logical"]
    total = total_2q + counts["meas_idle"]
    result = {"num_seq": len(seqs), "depths": DEPTHS_FO,
              "locations": dict(counts), "ambiguous": dict(ambiguous),
              "total_2q": total_2q, "total_all": total,
              "distinct_signatures_per_location": {
                  k: [int(min(v)), int(max(v))] for k, v in nsig.items()},
              "other_channels_if_heralded": {
                  k: [extra_amb[k], extra_counts[k]] for k in extra_counts}}
    print(f"circuit-level erasures: {total_2q} 2q locations "
          f"({counts['cx_check']} check CNOT + {counts['gate2_logical']} "
          f"logical SWAP/CZ), {counts['meas_idle']} measurement-idle; "
          f"total {total}; ambiguous {dict(ambiguous)}")
    print("  distinct detector signatures per location (min, max):",
          result["distinct_signatures_per_location"])
    print("  other channels (ambiguous, count):",
          result["other_channels_if_heralded"])

    # Hand-checkable example: first CNOT 7->0 of S_X1 in round 0, depth 2.
    lc = Q.build_lrb_circuit(seqs[0], 2, p, terminal="data5")
    sens = Sensitivity(lc.circuit)
    example = None
    for loc in sens.locs:
        if loc["name"] == "DEPOLARIZE2" and loc["targets"] == [7, 0]:
            example = loc
            break
    names = "IXYZ"
    rows = []
    for pa, pb in itertools.product(range(4), repeat=2):
        v = sens.pauli(example["tick"], {7: pa, 0: pb})
        dets = [lc.detector_keys[j] for j in np.flatnonzero(v[:-1])]
        rows.append({"pauli": f"{names[pa]}7*{names[pb]}0",
                     "detectors(round,stab)": dets, "flips_L": bool(v[-1])})
    result["example_cx70_round0_depth2"] = rows
    print("example: Pauli after the first S_X1 CNOT (7->0) of round 0, depth 2")
    for r in rows:
        print(f"  {r['pauli']:8s} dets={r['detectors(round,stab)']} "
              f"L={int(r['flips_L'])}")
    _dump(Path(args.out), "erasure_circuit.json", result)
    return result


# ---------------------------------------------------------------------------
# 3. First-order census (lead's DEM approach)
# ---------------------------------------------------------------------------

P_SMALL = 1e-6


def _only(component: str) -> Q.SI1000:
    zero = {name: 0.0 for name in Q.SI1000.__dataclass_fields__}
    zero[component] = getattr(Q.SI1000(), component)
    return Q.SI1000(**zero)


CENSUS_NOISE = {
    "total": Q.SI1000(),
    "only_2q": _only("two_qubit"),
    "only_meas_idle": _only("meas_idle"),
    "no_2q (=2q_r1)": Q.SI1000().scaled(two_qubit=0.0),
    "2q_r0.5": Q.SI1000().scaled(two_qubit=0.5),
    "2q_r0.9": Q.SI1000().scaled(two_qubit=0.1),
    "all_r0.5": Q.SI1000().scaled(two_qubit=0.5, meas_idle=0.5),
    "all_r0.9": Q.SI1000().scaled(two_qubit=0.1, meas_idle=0.1),
    "no_2q_no_measidle (=all_r1)": Q.SI1000().scaled(two_qubit=0.0,
                                                     meas_idle=0.0),
}


def _e1(dem: stim.DetectorErrorModel, p: float) -> tuple[float, float]:
    w: dict = defaultdict(lambda: [0.0, 0.0])
    for mech in M.dem_mechanisms(dem):
        w[mech.detectors][int(mech.flips_observable)] += mech.probability
    e1 = sum(min(a, b) for s, (a, b) in w.items() if s)
    e1 += w[()][1] if () in w else 0.0
    r1 = sum(a + b for s, (a, b) in w.items() if s)
    return e1 / p, r1 / p


def cmd_census(args) -> dict:
    out = {}
    for label, nseq, depths in (("6seq", 6, DEPTHS_FO),
                                ("30seq", 30, [10, 22])):
        seqs = _sequences(nseq)
        e1 = {k: np.zeros((nseq, len(depths))) for k in CENSUS_NOISE}
        for si, seq in enumerate(seqs):
            for di, depth in enumerate(depths):
                for name, noise in CENSUS_NOISE.items():
                    lc = Q.build_lrb_circuit(seq, depth, P_SMALL,
                                             terminal="data5", noise=noise)
                    e1[name][si, di] = _e1(lc.circuit.detector_error_model(),
                                           P_SMALL)[0]
        res = {"depths": depths, "num_seq": nseq, "E1_mean": {}, "E1_sd": {},
               "reduction_vs_total": {}, "slope_10_22": {}}
        i10, i22 = depths.index(10), depths.index(22)
        for name in CENSUS_NOISE:
            res["E1_mean"][name] = e1[name].mean(0).tolist()
            res["E1_sd"][name] = e1[name].std(0).tolist()
            rel = 1.0 - e1[name][:, i22] / e1["total"][:, i22]
            res["reduction_vs_total"][name] = [float(rel.mean()),
                                               float(rel.std() / np.sqrt(nseq))]
            slope = (e1[name][:, i22] - e1[name][:, i10]) / 12.0
            res["slope_10_22"][name] = [float(slope.mean()),
                                        float(slope.std() / np.sqrt(nseq))]
        # Naive additive estimate for the 2q-only heralding: E1(2q only)/E1(total)
        naive = e1["only_2q"][:, i22] / e1["total"][:, i22]
        res["naive_only2q_over_total_m22"] = [float(naive.mean()),
                                              float(naive.std() / np.sqrt(nseq))]
        naive_s = ((e1["only_2q"][:, i22] - e1["only_2q"][:, i10])
                   / (e1["total"][:, i22] - e1["total"][:, i10]))
        res["naive_only2q_over_total_slope"] = [float(naive_s.mean()),
                                                float(naive_s.std()
                                                      / np.sqrt(nseq))]
        sl = ((e1["no_2q (=2q_r1)"][:, i22] - e1["no_2q (=2q_r1)"][:, i10])
              / (e1["total"][:, i22] - e1["total"][:, i10]))
        res["herald2q_slope_ratio"] = [float(1 - sl.mean()),
                                       float(sl.std() / np.sqrt(nseq))]
        out[label] = res
        print(f"census {label}: depths {depths}")
        for name in CENSUS_NOISE:
            m = res["E1_mean"][name]
            r = res["reduction_vs_total"][name]
            s = res["slope_10_22"][name]
            print(f"  {name:28s} E1/p = " + ", ".join(f"{x:7.2f}" for x in m)
                  + f" | reduction at m=22 {100 * r[0]:6.2f}% +- {100 * r[1]:.2f}"
                  f" | slope {s[0]:.3f} +- {s[1]:.3f}")
        print(f"  naive E1(2q only)/E1(total) at m=22: "
              f"{100 * res['naive_only2q_over_total_m22'][0]:.2f}% ; "
              f"slope ratio {100 * res['naive_only2q_over_total_slope'][0]:.2f}%")
    _dump(Path(args.out), "census.json", out)
    return out


# ---------------------------------------------------------------------------
# 4. Flags: PauliString propagation
# ---------------------------------------------------------------------------


def _check_body(kind: str, flag_positions):
    """Gate list of one weight-3 check on qubits a=0, d=1,2,3, f=4.

    ``flag_positions``: indices i (0..3) where a flag CNOT is inserted before
    data CNOT i (3 = after the last). Returns (prep, body, post) where body
    is the list of 2q gates with labels.
    """
    a, f = 0, 4
    data = (1, 2, 3)
    body = []
    for i in range(4):
        for _ in range(list(flag_positions).count(i)):
            body.append(("F", ("CX", (f, a)) if kind == "Z" else ("CX", (a, f))))
        if i < 3:
            d = data[i]
            body.append((f"d{i + 1}", ("CX", (d, a)) if kind == "Z"
                         else ("CX", (a, d))))
    if kind == "X":
        prep, post = [("H", (a,))], [("H", (a,))]
    else:
        prep, post = [], []
    if flag_positions:
        if kind == "Z":
            prep = prep + [("H", (f,))]
            post = post + [("H", (f,))]
    return prep, body, post


def _propagate(pauli: stim.PauliString, gates) -> stim.PauliString:
    c = stim.Circuit()
    for name, targets in gates:
        c.append(name, list(targets))
    if len(c) == 0:
        return pauli
    return pauli.after(c)


def _fault_table(kind: str, flag_positions):
    prep, body, post = _check_body(kind, flag_positions)
    rows = []
    labels = [lab for lab, _ in body]
    fcount = 0
    for k, (lab, _) in enumerate(body):
        if lab == "F":
            fcount += 1
            lab = f"F{fcount}"
        for c in "XYZ":
            ps = stim.PauliString(5)
            ps[0] = c
            rest = [g for _, g in body[k + 1:]] + post
            fin = _propagate(ps, rest)
            data = "".join("_XYZ"[fin[q]] for q in (1, 2, 3))
            anc_flip = fin[0] in (1, 2)        # Z-basis measurement of a
            flag_flip = bool(flag_positions) and fin[4] in (1, 2)
            rows.append({"after": lab, "fault": f"{c}_a", "data_error": data,
                         "ancilla_meas_flip": anc_flip,
                         "flag_flip": flag_flip})
    return labels, rows


def cmd_flags(args) -> dict:
    result = {}
    layouts = {"none": (), "standard (d1 F d2 F d3)": (1, 2),
               "wrap (F d1 d2 F d3)": (0, 2)}
    for kind in ("X", "Z"):
        for lname, pos in layouts.items():
            labels, rows = _fault_table(kind, pos)
            result[f"{kind}-check | {lname}"] = rows
            print(f"{kind}-check, layout {lname}: order {' '.join(labels)}")
            for r in rows:
                spread = sum(ch != "_" for ch in r["data_error"])
                print(f"   fault {r['fault']} after {r['after']:3s}: data "
                      f"{r['data_error']} (weight {spread}) anc-flip "
                      f"{int(r['ancilla_meas_flip'])} flag {int(r['flag_flip'])}")
    # Which two-CNOT placements flag every ancilla fault that spreads to data?
    scan = {}
    for kind in ("X", "Z"):
        for pos in itertools.combinations_with_replacement(range(4), 2):
            _, rows = _fault_table(kind, pos)
            spreading = [r for r in rows if r["data_error"] != "___"
                         and not r["after"].startswith("F")]
            unflagged = [f"{r['fault']}@{r['after']}" for r in spreading
                         if not r["flag_flip"]]
            scan[f"{kind}{pos}"] = unflagged
    result["placement_scan_unflagged_spreading_faults"] = scan
    print("placement scan (flag CNOTs inserted before data CNOT index i; "
          "3 = after the last): unflagged spreading faults after data CNOTs")
    for k, v in scan.items():
        print(f"   {k}: {v}")
    _dump(Path(args.out), "flags_propagation.json", result)
    return result


def cmd_flag_e1(args) -> dict:
    """E1 of the implementer's flagged circuits from Stim's DEM."""
    from lrb_qubit import erasure as ER  # the object under test
    seqs = _sequences(args.num_seq)
    depths = [int(d) for d in args.depths.split(",")] if args.depths else [
        0, 2, 6, 10]
    configs = {"none": (None, "standard"),
               "ideal_standard": ("ideal", "standard"),
               "ideal_wrap": ("ideal", "wrap"),
               "noisy_standard": ("noisy", "standard"),
               "noisy_wrap": ("noisy", "wrap")}
    e1 = {k: np.zeros((len(seqs), len(depths))) for k in configs}
    for si, seq in enumerate(seqs):
        for di, depth in enumerate(depths):
            for name, (flags, layout) in configs.items():
                lc = ER.build_instrumented_lrb(seq, depth, P_SMALL,
                                               flags=flags, flag_layout=layout,
                                               mode="decode")
                e1[name][si, di] = _e1(lc.circuit.detector_error_model(),
                                       P_SMALL)[0]
    res = {"depths": depths, "num_seq": len(seqs)}
    for name in configs:
        rel = e1[name] / e1["none"] - 1.0
        res[name] = {"E1_mean": e1[name].mean(0).tolist(),
                     "rel_change_mean": rel.mean(0).tolist(),
                     "rel_change_se": (rel.std(0) / np.sqrt(len(seqs))).tolist()}
        print(f"  {name:15s} E1/p " + ", ".join(
            f"{x:7.2f}" for x in res[name]["E1_mean"]) + " | rel change "
            + ", ".join(f"{100 * x:+.2f}%" for x in res[name]["rel_change_mean"]))
    _dump(Path(args.out), "flag_e1.json", res)
    return res


# ---------------------------------------------------------------------------
# 5. Exact first-order effect of one leak event (own GF(2) analysis)
# ---------------------------------------------------------------------------


def _record_signatures(circuit: stim.Circuit) -> np.ndarray:
    """recsig[r] = targets (dets + L0) that include measurement record r."""
    n_det = circuit.num_detectors
    sig = np.zeros((circuit.num_measurements, n_det + 1), dtype=bool)
    meas = 0
    det = 0
    for inst in circuit.flattened():
        if inst.name in ("M", "MR", "MX", "MY", "MZ"):
            meas += len(inst.targets_copy())
        elif inst.name == "DETECTOR":
            for t in inst.targets_copy():
                sig[meas + t.value, det] ^= True
            det += 1
        elif inst.name == "OBSERVABLE_INCLUDE":
            if int(inst.gate_args_copy()[0]) == 0:
                for t in inst.targets_copy():
                    sig[meas + t.value, n_det] ^= True
    return sig


def _timeline(circuit: stim.Circuit, sens: Sensitivity,
              include_fused: bool = True):
    """Events in time order: noisy 2q gates (pair, tick), measurements
    (qubit, record), resets (qubit) and reset-layer starts (tick).

    At depth 0 Stim fuses the noiseless encoder CX with the first check CNOT
    (``CX 2 1 1 0 3 0 4 2 0 5`` followed by ``DEPOLARIZE2 0 5``). With
    ``include_fused`` the noisy pair (0, 5) is kept (physical model);
    without it the gate is skipped, which mirrors WP I's simulator
    (``leakage._noisy_two_qubit_instructions`` compares full target lists).
    """
    tick_of = {loc["index"]: loc["tick"] for loc in sens.locs}
    instrs = list(circuit)
    events = []
    record = 0
    for i, inst in enumerate(instrs):
        name = inst.name
        targets = [t.value for t in inst.targets_copy()]
        if name in TWO_Q:
            nxt = instrs[i + 1] if i + 1 < len(instrs) else None
            if nxt is None or nxt.name != "DEPOLARIZE2":
                continue
            noisy = [t.value for t in nxt.targets_copy()]
            pairs = list(zip(targets[::2], targets[1::2]))
            if noisy == targets:
                keep = pairs
            elif include_fused:
                npairs = set(zip(noisy[::2], noisy[1::2]))
                keep = [pr for pr in pairs if pr in npairs]
            else:
                keep = []
            for a, b in keep:
                events.append(("gate", (a, b), tick_of[i + 1]))
        elif name == "M":
            for q in targets:
                events.append(("meas", q, record))
                record += 1
        elif name == "R":
            if targets and targets[0] == 5:
                # LRU acts here (before the reset); data sensitivity equals
                # that at the TICK after the reset-idle noise just before.
                prev = [loc for loc in sens.locs if loc["index"] < i]
                events.append(("layer", None, prev[-1]["tick"] if prev
                               else None))
            for q in targets:
                events.append(("reset", q, None))
    return events


def _gf2_accept(components: list[np.ndarray], offset: np.ndarray,
                check_map) -> tuple[float, float]:
    """P(accept) and P(accept & L flip) for offset + uniform span(components).

    ``check_map(v)`` maps a (dets + L0) vector to (checks bit-vector, L bit).
    """
    def to_int(v):
        checks, lbit = check_map(v)
        val = 0
        for j in np.flatnonzero(checks):
            val |= 1 << int(j)
        return val, int(lbit)

    basis: dict[int, tuple[int, int]] = {}
    obs_in_span = False
    for comp in components:
        w, lw = to_int(comp)
        while w:
            piv = w.bit_length() - 1
            if piv in basis:
                bw, bl = basis[piv]
                w ^= bw
                lw ^= bl
            else:
                basis[piv] = (w, lw)
                break
        if not w and lw:
            obs_in_span = True
    w, lw = to_int(offset)
    while w:
        piv = w.bit_length() - 1
        if piv not in basis:
            return 0.0, 0.0
        bw, bl = basis[piv]
        w ^= bw
        lw ^= bl
    p_acc = 2.0 ** (-len(basis))
    p_flip = 0.5 if obs_in_span else float(lw)
    return p_acc, p_acc * p_flip


def leak_first_order(lc: Q.LRBCircuit, lru: bool = False,
                     readout_value: int = 1,
                     include_fused: bool = True) -> dict:
    circuit = lc.circuit
    sens = Sensitivity(circuit)
    recsig = _record_signatures(circuit)
    ref = circuit.reference_sample()
    events = _timeline(circuit, sens, include_fused)
    n_det = circuit.num_detectors
    n_round = 4 * lc.num_rounds
    terminal_pos = lc.rounds.index(lc.depth)

    # const=1 accepts iff the terminal round's raw syndromes vanish; raw
    # syndromes are a GF(2)-linear function of the round detectors.
    eye = np.eye(n_round, dtype=bool)
    A = lc.raw_syndromes(eye)[:, terminal_pos, :].astype(np.int64)

    def map_u1f(v):
        return v[:n_det], v[n_det]

    def map_const1(v):
        return (v[:n_round].astype(np.int64) @ A) % 2 == 1, v[n_det]

    totals = {"u1f": [0.0, 0.0], "const1": [0.0, 0.0]}
    by_source = defaultdict(lambda: [0.0, 0.0])
    n_events = 0
    per_event = []
    gate_events = [(k, e) for k, e in enumerate(events) if e[0] == "gate"]
    for k, (_, pair, tick) in gate_events:
        for leaker in pair:
            partner = pair[1] if leaker == pair[0] else pair[0]
            comps = [sens.component(tick, partner, "X"),
                     sens.component(tick, partner, "Z")]
            offset = np.zeros(n_det + 1, dtype=bool)
            for kind, what, extra in events[k + 1:]:
                if kind == "gate" and leaker in what:
                    other = what[1] if what[0] == leaker else what[0]
                    comps += [sens.component(extra, other, "X"),
                              sens.component(extra, other, "Z")]
                elif kind == "meas" and what == leaker:
                    if leaker >= 5:   # ancilla: forced to readout_value
                        if readout_value ^ int(ref[extra]):
                            offset ^= recsig[extra]
                    else:             # data, terminal X readout: random flip
                        comps.append(recsig[extra].copy())
                elif kind == "reset" and what == leaker:
                    break
                elif kind == "layer" and lru and leaker < 5:
                    comps += [sens.component(extra, leaker, "X"),
                              sens.component(extra, leaker, "Z")]
                    break
            n_events += 1
            res = {}
            for name, cmap in (("u1f", map_u1f), ("const1", map_const1)):
                pa, pf = _gf2_accept(comps, offset, cmap)
                totals[name][0] += pa
                totals[name][1] += pf
                res[name] = (pa, pf)
            src = ("anc" if leaker >= 5 else "data") + (
                "@check" if max(pair) >= 5 else "@logical")
            by_source[src][0] += res["u1f"][0]
            by_source[src][1] += res["u1f"][1]
            per_event.append((tick, leaker, partner, res["u1f"][1]))
    return {"num_leak_events": n_events, "num_gates": len(gate_events),
            "C_acc": {k: v[0] for k, v in totals.items()},
            "C_acc_flip": {k: v[1] for k, v in totals.items()},
            "u1f_by_source": {k: list(v) for k, v in by_source.items()},
            "top_events": sorted(per_event, key=lambda x: -x[3])[:12]}


def cmd_leakfo(args) -> dict:
    seqs = _sequences(args.num_seq)
    depths = [int(d) for d in args.depths.split(",")] if args.depths else [
        0, 2, 4, 10, 22]
    out = {}
    for lru in (False, True):
        rows = []
        for si, seq in enumerate(seqs):
            for depth in depths:
                lc = Q.build_lrb_circuit(seq, depth, 1e-3, terminal="data5")
                r = leak_first_order(lc, lru=lru)
                rows.append({"seq": si, "depth": depth, **{
                    k: r[k] for k in ("num_leak_events", "C_acc",
                                      "C_acc_flip", "u1f_by_source")}})
                if si == 0 and depth == depths[-1]:
                    out[f"top_events_lru{int(lru)}_seq0_m{depth}"] = r[
                        "top_events"]
        summary = {}
        for depth in depths:
            sel = [r for r in rows if r["depth"] == depth]
            summary[depth] = {
                "events": float(np.mean([r["num_leak_events"] for r in sel])),
                "C_acc_flip_u1f": [float(np.mean([r["C_acc_flip"]["u1f"]
                                                  for r in sel])),
                                   float(np.std([r["C_acc_flip"]["u1f"]
                                                 for r in sel]))],
                "C_acc_u1f": float(np.mean([r["C_acc"]["u1f"] for r in sel])),
                "C_acc_flip_const1": float(np.mean([r["C_acc_flip"]["const1"]
                                                    for r in sel])),
                "C_acc_const1": float(np.mean([r["C_acc"]["const1"]
                                               for r in sel])),
                "u1f_by_source": {s: float(np.mean([r["u1f_by_source"].get(
                    s, [0, 0])[1] for r in sel])) for s in (
                    "data@check", "anc@check", "data@logical")}}
        out[f"lru={lru}"] = {"rows": rows, "summary": summary}
        print(f"leak first order, LRU={lru} ({len(seqs)} sequences); "
              "coefficients per unit kappa*p:")
        for depth, s in summary.items():
            print(f"  m={depth:2d}: leak events {s['events']:.0f}; u1f "
                  f"P(acc&flip) sum C = {s['C_acc_flip_u1f'][0]:.4f} "
                  f"(sd {s['C_acc_flip_u1f'][1]:.4f}), P(acc) sum "
                  f"{s['C_acc_u1f']:.3f}; const1 C = "
                  f"{s['C_acc_flip_const1']:.3f}; by source "
                  f"{s['u1f_by_source']}")
    _dump(Path(args.out), "leak_first_order.json", out)
    return out


# ---------------------------------------------------------------------------
# 6. Leakage tables from WP I raw parts
# ---------------------------------------------------------------------------

PROBS = [5e-4, 2e-3, 5e-3]
LEAK_CONFIGS = ["none", "k0.1", "k0.1_lru", "k0.5", "k0.5_lru"]
KAPPA = {"none": 0.0, "k0.1": 0.1, "k0.1_lru": 0.1, "k0.5": 0.5,
         "k0.5_lru": 0.5}


def _load(prefix: str, n: int = 12):
    files = [IRES / "leakage" / "parts" / f"{prefix}_s{i}.npz"
             for i in range(n)]
    datas = [np.load(f) for f in files if f.exists()]
    if len(datas) != n:
        raise FileNotFoundError(prefix)
    tables = {}
    for name in sorted({k.split("__")[0] for k in datas[0].files
                        if k.endswith("__shots")}):
        tables[name] = (np.stack([d[f"{name}__shots"] for d in datas]),
                        np.stack([d[f"{name}__accepted"] for d in datas]),
                        np.stack([d[f"{name}__errors"] for d in datas]))
    return tables, datas, [int(x) for x in datas[0]["depths"]]


def _fid(t):
    shots, acc, err = t
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.abs(1 - 2 * err / acc), acc / shots


def _local_rates(Rm, depths, pairs):
    return [(Rm[..., depths.index(b)] / Rm[..., depths.index(a)])
            ** (1.0 / (b - a)) for a, b in pairs]


def cmd_leaktables(args) -> dict:
    rng = np.random.default_rng(12345)
    pairs = [(0, 4), (4, 10), (10, 22)]
    out = {"L4": {}, "curvature": {}, "u1f": {}, "mld_flag": {},
           "const1": {}, "trajectory": {}, "lru_rate_reduction": {}}
    seqs = _sequences(12)
    for pi, p in enumerate(PROBS):
        base, _, depths = _load(f"none_p{pi}")
        F0 = {m: _fid(base[m])[0] for m in base}
        for cfg in LEAK_CONFIGS[1:]:
            t, datas, _ = _load(f"{cfg}_p{pi}")
            F1 = {m: _fid(t[m])[0] for m in t}
            for method in ("mld", "lrbd_const_1"):
                a, b = F1[method], F0[method]
                R = np.nanmean(a, 0) / np.nanmean(b, 0)
                rates = [float(x) for x in _local_rates(R, depths, pairs)]
                boots = []
                for _ in range(2000):
                    idx = rng.integers(0, a.shape[0], a.shape[0])
                    Rb = np.nanmean(a[idx], 0) / np.nanmean(b[idx], 0)
                    boots.append(_local_rates(Rb, depths, pairs))
                boots = np.array(boots)
                se = np.nanstd(boots, axis=0).tolist()
                key = f"{p:g}|{cfg}|{method}"
                out["L4"][key] = {
                    "R": {d: float(R[depths.index(d)]) for d in (4, 10, 22)},
                    "rates": rates, "rates_se": se,
                    "rate_late_minus_early": [
                        float(rates[2] - rates[0]),
                        float(np.nanstd(boots[:, 2] - boots[:, 0]))]}
                # curvature of log R over all depths (weighted quadratic fit)
                if method == "mld":
                    d = np.asarray(depths, float)
                    good = np.isfinite(R) & (R > 0)

                    def quad(Rv):
                        y = np.log(Rv[good])
                        X = np.stack([np.ones(good.sum()), d[good],
                                      d[good] ** 2], 1)
                        return np.linalg.lstsq(X, y, rcond=None)[0]
                    coef = quad(R)
                    bc = []
                    for _ in range(1000):
                        idx = rng.integers(0, a.shape[0], a.shape[0])
                        Rb = np.nanmean(a[idx], 0) / np.nanmean(b[idx], 0)
                        if np.all(np.isfinite(Rb[good])) and np.all(Rb[good] > 0):
                            bc.append(quad(Rb))
                    bc = np.array(bc)
                    out["curvature"][key] = {
                        "slope_per_step": [float(coef[1]),
                                           float(bc[:, 1].std())],
                        "quad_coef": [float(coef[2]), float(bc[:, 2].std())],
                        "rate_change_over_0_22": [
                            float(np.exp(2 * coef[2] * 22) - 1),
                            float(2 * 22 * bc[:, 2].std())]}
            # LRB-D u1f under leakage, paired by sequence (independent shots)
            fu1, au1 = _fid(t["lrbd_unif_1_final"])
            fu0, au0 = _fid(base["lrbd_unif_1_final"])
            diff = fu1 - fu0
            n = np.sum(np.isfinite(diff), 0)
            dm = np.nanmean(diff, 0)
            dse = np.nanstd(diff, 0) / np.sqrt(np.maximum(n, 1))
            # pooled accepted error rates
            e1_ = t["lrbd_unif_1_final"][2].sum(0) / t["lrbd_unif_1_final"][1].sum(0)
            e0_ = base["lrbd_unif_1_final"][2].sum(0) / base["lrbd_unif_1_final"][1].sum(0)
            se1 = np.sqrt(e1_ / t["lrbd_unif_1_final"][1].sum(0))
            se0 = np.sqrt(e0_ / base["lrbd_unif_1_final"][1].sum(0))
            out["u1f"][f"{p:g}|{cfg}"] = {
                "depths": depths, "dF_mean": dm.tolist(), "dF_se": dse.tolist(),
                "z": (dm / dse).tolist(),
                "eps_leak": e1_.tolist(), "eps_none": e0_.tolist(),
                "eps_leak_se": se1.tolist(), "eps_none_se": se0.tolist(),
                "delta_eps": (e1_ - e0_).tolist(),
                "delta_eps_se": np.sqrt(se1 ** 2 + se0 ** 2).tolist(),
                "acc_ratio": (np.nanmean(au1, 0) / np.nanmean(au0, 0)).tolist()}
            # u1f plus leakage-flag rejection (does it remove the excess?)
            tf = t["lrbd_unif_1_final_flag"]
            ef_ = tf[2].sum(0) / tf[1].sum(0)
            out["u1f"][f"{p:g}|{cfg}"].update({
                "eps_leak_flag": ef_.tolist(),
                "eps_leak_flag_se": np.sqrt(ef_ / tf[1].sum(0)).tolist()})
            # lag >= 2 detection autocorrelation (persistence fingerprint)
            ac = np.mean([dd["autocorr"] for dd in datas], axis=0).mean(axis=1)
            ac0 = np.mean([dd["autocorr"] for dd in _load(f"none_p{pi}")[1]],
                          axis=0).mean(axis=1)
            out.setdefault("autocorr", {})[f"{p:g}|{cfg}"] = {
                "leak": ac.tolist(), "none": ac0.tolist()}
            # mld_flag vs none, and the "perfect flags" prediction
            fmf, amf = _fid(t["mld_flag"])
            out["mld_flag"][f"{p:g}|{cfg}"] = {
                "F_mld_flag": np.nanmean(fmf, 0).tolist(),
                "F_mld_none": np.nanmean(F0["mld"], 0).tolist(),
                "acc_mld_flag": np.nanmean(amf, 0).tolist()}
            fc, ac = _fid(t["lrbd_const_1"])
            fcf, acf = _fid(t["lrbd_const_1_flag"])
            out["const1"][f"{p:g}|{cfg}"] = {
                "F": np.nanmean(fc, 0).tolist(),
                "F_none": np.nanmean(F0["lrbd_const_1"], 0).tolist(),
                "F_flag": np.nanmean(fcf, 0).tolist(),
                "acc_flag": np.nanmean(acf, 0).tolist()}
            traj = np.mean([dd["leaked_trajectory"] for dd in datas], axis=0)
            out["trajectory"][f"{p:g}|{cfg}"] = traj.tolist()
    # Perfect-flag and geometric-trajectory predictions from circuit counts.
    pred = {}
    for pi, p in enumerate(PROBS):
        for cfg in ("k0.1", "k0.5"):
            kp = KAPPA[cfg] * p
            acc_pred = []
            traj_pred = []
            for seq in seqs:
                lc = Q.build_lrb_circuit(seq, 22, p, terminal="data5")
                circ = list(lc.circuit)
                n_part = 0          # all qubit participations in noisy 2q gates
                n_data = 0          # data participations
                before_layers = []
                for i, inst in enumerate(circ):
                    tg = [t.value for t in inst.targets_copy()]
                    if inst.name in TWO_Q and i + 1 < len(circ) and \
                            circ[i + 1].name == "DEPOLARIZE2":
                        n_part += len(tg)
                        n_data += sum(1 for q in tg if q < 5)
                    if inst.name == "R" and tg and tg[0] == 5:
                        before_layers.append(n_data)
                before_layers.append(n_data)  # terminal readout
                acc_pred.append((1 - kp) ** n_part)
                traj_pred.append([1 - (1 - kp) ** n for n in before_layers])
            pred[f"{p:g}|{cfg}"] = {
                "P_no_leak_m22": float(np.mean(acc_pred)),
                "trajectory_geometric": np.mean(traj_pred, axis=0).tolist()}
    out["predictions"] = pred
    # LRU rate reduction from the local rates (mean of the three windows)
    for p in PROBS:
        for k in ("k0.1", "k0.5"):
            a = np.mean(out["L4"][f"{p:g}|{k}|mld"]["rates"])
            b = np.mean(out["L4"][f"{p:g}|{k}_lru|mld"]["rates"])
            out["lru_rate_reduction"][f"{p:g}|{k}"] = float(
                1 - np.log(b) / np.log(a))
    # printing
    print("Table L4 recomputed (local rates of R = F_leak/F_none, bootstrap SE):")
    for key, v in out["L4"].items():
        print(f"  {key:28s} R(4,10,22) = " + ", ".join(
            f"{v['R'][d]:.4f}" for d in (4, 10, 22)) + " | rates " + ", ".join(
            f"{r:.4f}({1e4 * s:.0f})" for r, s in zip(v["rates"], v["rates_se"]))
            + f" | late-early {v['rate_late_minus_early'][0]:+.4f}"
              f" +- {v['rate_late_minus_early'][1]:.4f}")
    print("curvature of log R(m) (quadratic fit over all depths):")
    for key, v in out["curvature"].items():
        print(f"  {key:22s} slope {v['slope_per_step'][0]:+.5f}"
              f"({1e5 * v['slope_per_step'][1]:.0f}) quad "
              f"{v['quad_coef'][0]:+.2e} +- {v['quad_coef'][1]:.1e}")
    print("LRU reduction of the leakage-induced rate:",
          {k: round(v, 3) for k, v in out["lru_rate_reduction"].items()})
    print("LRB-D unif=1+final under leakage (accepted error eps, paired dF):")
    for key, v in out["u1f"].items():
        for di, d in enumerate(v["depths"]):
            if d in (0, 4, 10, 22):
                print(f"  {key:14s} m={d:2d}: eps {v['eps_none'][di]:.2e} -> "
                      f"{v['eps_leak'][di]:.2e} (delta {v['delta_eps'][di]:+.2e}"
                      f" +- {v['delta_eps_se'][di]:.1e}); dF "
                      f"{v['dF_mean'][di]:+.5f} +- {v['dF_se'][di]:.5f}"
                      f" (z {v['z'][di]:+.1f}); acc ratio {v['acc_ratio'][di]:.3f}"
                      f"; +flag eps {v['eps_leak_flag'][di]:.2e}")
    print("lag 1..6 detection autocorrelation (mean over stabilizers):")
    for key, v in out["autocorr"].items():
        print(f"  {key:14s} leak " + ", ".join(f"{x:.4f}" for x in v["leak"][1:])
              + " | none " + ", ".join(f"{x:.4f}" for x in v["none"][1:]))
    print("mld_flag vs none and perfect-flag acceptance prediction (m=22):")
    for key, v in out["mld_flag"].items():
        pk = key.replace("_lru", "")
        print(f"  {key:14s} F mld_flag {v['F_mld_flag'][-1]:.4f} vs none "
              f"{v['F_mld_none'][-1]:.4f}; acc {v['acc_mld_flag'][-1]:.4f} vs "
              f"predicted P(no leak) {pred[pk]['P_no_leak_m22']:.4f}")
    print("const=1 at m=22: F none -> leak -> leak+flag (acc flag):")
    for key, v in out["const1"].items():
        print(f"  {key:14s} {v['F_none'][-1]:.4f} -> {v['F'][-1]:.4f} -> "
              f"{v['F_flag'][-1]:.4f} ({v['acc_flag'][-1]:.4f})")
    print("leaked-data population (rounds 1, 5, 10, 21, readout) vs "
          "1-(1-kp)^N(k):")
    for key, v in out["trajectory"].items():
        if "lru" in key:
            continue
        g = pred[key]["trajectory_geometric"]
        idx = [1, 5, 10, 21, 22]
        lin = [v[1] / 2 * (k + 1) for k in (1, 5, 10, 21)]
        print(f"  {key:10s} sim " + ", ".join(f"{v[i]:.4f}" for i in idx)
              + " | geometric " + ", ".join(f"{g[i]:.4f}" for i in idx)
              + " | linear-from-round-1 " + ", ".join(f"{x:.4f}" for x in lin))
    _dump(Path(args.out), "leak_tables.json", out)
    return out


# ---------------------------------------------------------------------------
# 6b. Erasure tables from WP I raw parts (claims 7-10)
# ---------------------------------------------------------------------------

ECONFIGS = ["r0", "2q_r0.5", "2q_r0.9", "2q_r1", "all_r0.5", "all_r0.9",
            "all_r1"]


def _load_e(prefix: str, n: int = 12):
    files = [IRES / "erasure" / "parts" / f"{prefix}_s{i}.npz"
             for i in range(n)]
    datas = [np.load(f) for f in files]
    get = {}
    for key in datas[0].files:
        if "__" in key and not key.startswith("gap__"):
            get[key] = np.stack([d[key] for d in datas])
    hist = np.stack([d["gap__mld_herald"] for d in datas])
    return get, hist, [int(x) for x in datas[0]["depths"]], datas[0][
        "gap_edges"]


def _fit_loglin(depths, F, rng, boots=300):
    """f from a weighted LS fit of ln(mean F) = a + m ln f (own fit)."""
    d = np.asarray(depths, float)

    def fit(Fs):
        mean = np.nanmean(Fs, 0)
        se = np.nanstd(Fs, 0) / np.sqrt(Fs.shape[0]) + 1e-4
        keep = (mean > 0.02) & (mean < 1.0) & np.isfinite(mean)
        y = np.log(mean[keep])
        w = (mean[keep] / se[keep]) ** 2
        X = np.stack([np.ones(keep.sum()), d[keep]], 1)
        coef = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
        return float(np.exp(coef[1]))

    f0 = fit(F)
    bs = [fit(F[rng.integers(0, F.shape[0], F.shape[0])]) for _ in range(boots)]
    return f0, float(np.std(bs))


def cmd_erasuretables(args) -> dict:
    rng = np.random.default_rng(7)
    out = {"fits": {}, "hybrid": {}, "gap": {}}
    for pi, p in enumerate(PROBS):
        for cfg in ECONFIGS:
            g, hist, depths, edges = _load_e(f"{cfg}_p{pi}")

            def F(method):
                acc = g[f"{method}__accepted"].astype(float)
                err = g[f"{method}__errors"].astype(float)
                with np.errstate(invalid="ignore", divide="ignore"):
                    return np.abs(1 - 2 * err / acc)
            fm = _fit_loglin(depths, F("mld"), rng)
            fh = _fit_loglin(depths, F("mld_herald"), rng)
            out["fits"][f"{p:g}|{cfg}"] = {"f_mld": fm, "f_mld_herald": fh,
                                           "eps_reduction": 1 - (1 - fh[0])
                                           / (1 - fm[0])}
            for di, m in enumerate(depths):
                if m not in (4, 10):
                    continue
                pooled = {}
                for method in ("lrbd_unif_1_final", "hybrid",
                               "lrbd_unif_1_final_noherald"):
                    a = g[f"{method}__accepted"][:, di].sum()
                    e = g[f"{method}__errors"][:, di].sum()
                    s = g[f"{method}__shots"][:, di].sum()
                    pooled[method] = (a / s, 1 - 2 * e / a if a else np.nan,
                                      e / a if a else np.nan)
                out["hybrid"][f"{p:g}|{cfg}|m{m}"] = {
                    "acc_ratio": pooled["hybrid"][0]
                    / pooled["lrbd_unif_1_final"][0],
                    "F_hybrid": pooled["hybrid"][1],
                    "F_u1f": pooled["lrbd_unif_1_final"][1],
                    "eps_ratio": pooled["hybrid"][2]
                    / pooled["lrbd_unif_1_final"][2],
                    "noherald_acc_ratio": pooled[
                        "lrbd_unif_1_final_noherald"][0]
                    / pooled["lrbd_unif_1_final"][0],
                    "F_noherald": pooled["lrbd_unif_1_final_noherald"][1]}
                # herald-aware |LLR| postselection at exactly LRB-D's
                # acceptance (pooled, fractional last bin)
                h = hist[:, di].sum(axis=0)          # (bins, 2)
                total = h.sum()
                target = pooled["lrbd_unif_1_final"][0] * total
                acc = err = 0.0
                thr = None
                for b in range(h.shape[0] - 1, -1, -1):
                    nb = h[b].sum()
                    if acc + nb >= target:
                        frac = (target - acc) / nb if nb else 0.0
                        acc += frac * nb
                        err += frac * h[b, 1]
                        thr = float(edges[b])
                        break
                    acc += nb
                    err += h[b, 1]
                out["gap"][f"{p:g}|{cfg}|m{m}"] = {
                    "threshold_bin_low_edge": thr,
                    "acc": acc / total, "F_gap": 1 - 2 * err / acc,
                    "F_u1f": pooled["lrbd_unif_1_final"][1]}
    print("fitted f (own log-linear WLS, bootstrap SE):")
    for k, v in out["fits"].items():
        if k.startswith("0.002") or k.endswith("r1"):
            print(f"  {k:16s} mld {v['f_mld'][0]:.4f}({1e4 * v['f_mld'][1]:.0f})"
                  f"  mld_herald {v['f_mld_herald'][0]:.4f}"
                  f"({1e4 * v['f_mld_herald'][1]:.0f})  error/step reduction "
                  f"{100 * v['eps_reduction']:.1f}%")
    print("hybrid vs LRB-D u1f (pooled):")
    for k, v in out["hybrid"].items():
        if "|r0|" in k:
            continue
        print(f"  {k:20s} acc x{v['acc_ratio']:.2f}  F {v['F_hybrid']:.4f} vs "
              f"{v['F_u1f']:.4f}  accepted-error ratio {v['eps_ratio']:.2f}"
              f" | u1f+noherald acc x{v['noherald_acc_ratio']:.2f} "
              f"F {v['F_noherald']:.4f}")
    print("herald-aware gap postselection at exactly LRB-D acceptance:")
    for k, v in out["gap"].items():
        print(f"  {k:20s} acc {v['acc']:.4f}  F gap {v['F_gap']:.4f} vs "
              f"LRB-D {v['F_u1f']:.4f}")
    _dump(Path(args.out), "erasure_tables.json", out)
    return out


def cmd_heraldcal(args) -> dict:
    """Calibration of the implementer's HeraldedMLD on the TRUE heralded
    channel at configurations its own test does not use (depth 2 and 4,
    two-qubit AND measurement-idle heralding, r = 0.5 and 1): reliability
    bins of the posterior P(L=1 | s, h) against empirical frequencies."""
    from lrb_qubit import erasure as ER  # object under test
    seq = _sequences(3)[2]
    res = {}
    for depth, r, p in ((2, 1.0, 0.02), (4, 0.5, 0.01)):
        spec = ER.ErasureSpec(two_qubit=r, meas_idle=r)
        ec = ER.prepare_erasure_circuits(seq, depth, p, spec)
        n = args.cal_shots
        events, obs = ec.sample.circuit.compile_detector_sampler(
            seed=args.cal_seed + depth).sample(n, separate_observables=True)
        dets, heralds = ec.sample.split(events)
        llr = ec.mld_herald.log_likelihood_ratio(dets, heralds)
        post = 1.0 / (1.0 + np.exp(np.clip(llr, -700, 700)))  # P(L=1|s,h)
        flip = obs[:, 0]
        wrong = (llr < 0) != flip
        pred_err = np.minimum(post, 1 - post).mean()
        edges = np.array([0, 1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.2, 0.3, 0.4,
                          0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999, 0.9999,
                          1.0 + 1e-12])
        idx = np.clip(np.searchsorted(edges, post, side="right") - 1, 0,
                      len(edges) - 2)
        chi2 = 0.0
        dof = 0
        rows = []
        for b in range(len(edges) - 1):
            sel = idx == b
            m = int(sel.sum())
            if m < 50:
                continue
            exp1 = post[sel].sum()
            var = (post[sel] * (1 - post[sel])).sum()
            k = int(flip[sel].sum())
            if var > 1.0:
                chi2 += (k - exp1) ** 2 / var
                dof += 1
            rows.append([float(edges[b]), m, k, float(exp1)])
        res[f"depth{depth}_r{r}_p{p}"] = {
            "shots": n, "heralds_per_shot": float(heralds.sum(1).mean()),
            "empirical_error": float(wrong.mean()),
            "predicted_error": float(pred_err),
            "se": float(np.sqrt(pred_err * (1 - pred_err) / n)),
            "chi2": chi2, "dof": dof, "bins": rows}
        v = res[f"depth{depth}_r{r}_p{p}"]
        print(f"  depth {depth}, r={r}, p={p}: heralds/shot "
              f"{v['heralds_per_shot']:.2f}; error {v['empirical_error']:.5f}"
              f" vs predicted {v['predicted_error']:.5f} (+-{v['se']:.5f});"
              f" reliability chi2 {chi2:.1f} / {dof} bins")
    _dump(Path(args.out), "herald_calibration.json", res)
    return res


# ---------------------------------------------------------------------------
# 7. New-seed simulations
# ---------------------------------------------------------------------------


def _u1f_counts(lc, sample):
    n_round = 4 * lc.num_rounds
    raw = lc.raw_syndromes(sample.dets[:, :n_round])
    final_clean = ~sample.dets[:, n_round:].any(axis=1)
    acc = B.accept_mask(raw, lc.rounds, B.used_rounds("unif", 1, lc.depth)) \
        & final_clean
    accf = acc & ~sample.any_flag()
    return (int(acc.sum()), int((acc & sample.obs).sum()),
            int(accf.sum()), int((accf & sample.obs).sum()))


def _leaksim_task(task):
    from lrb_qubit import leakage as L
    p, kappa, lru, depth, si, shots, seed = task
    seq = _sequences(si + 1)[si]
    lc = Q.build_lrb_circuit(seq, depth, p, terminal="data5")
    counts = np.zeros(4, dtype=np.int64)
    done = 0
    while done < shots:
        n = min(50_000, shots - done)
        s = L.simulate_leakage(lc, p, L.LeakageModel(rate=kappa, lru=lru), n,
                               seed=seed + done)
        counts += np.array(_u1f_counts(lc, s))
        done += n
    return task, counts


def _const1_task(task):
    p, depth, si, shots, seed = task
    seq = _sequences(si + 1)[si]
    lc = Q.build_lrb_circuit(seq, depth, p, terminal="data5")
    sampler = lc.circuit.compile_detector_sampler(seed=seed)
    dets, obs = sampler.sample(shots, separate_observables=True)
    raw = lc.raw_syndromes(dets[:, :4 * lc.num_rounds])
    acc = B.accept_mask(raw, lc.rounds, B.used_rounds("const", 1, depth))
    return task, int(acc.sum()), int((acc & obs[:, 0]).sum())


def cmd_leaksim(args) -> dict:
    from concurrent.futures import ProcessPoolExecutor
    out = {}
    tasks = []
    shots = args.shots or 200_000
    # m = 22, p = 5e-4: kappa 0, 0.5, 0.5 + LRU; 12 sequences (new seeds)
    for si in range(args.num_seq):
        for kappa, lru in ((0.0, False), (0.5, False), (0.5, True)):
            tasks.append((5e-4, kappa, lru, 22, si, shots,
                          500_009 + 7919 * si + 31 * int(10 * kappa)
                          + int(lru)))
        for kappa in (0.0, 0.5):   # lower p: first order should dominate
            tasks.append((2.5e-4, kappa, False, 22, si, shots,
                          600_011 + 7919 * si + 31 * int(10 * kappa)))
    # m = 0 (one circuit): p = 2e-3 and 5e-4, many shots
    for p in (2e-3, 5e-4):
        for kappa in (0.0, 0.5):
            for part in range(8):
                tasks.append((p, kappa, False, 0, 0, shots,
                              900_001 + 104_729 * part + int(10 * kappa)
                              + int(p * 1e6)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_leaksim_task, tasks))
    agg = defaultdict(lambda: np.zeros(4, dtype=np.int64))
    for (p, kappa, lru, depth, si, n, seed), counts in results:
        agg[(p, kappa, lru, depth)] += counts
    rows = {}
    for (p, kappa, lru, depth), c in sorted(agg.items()):
        acc, err, accf, errf = (int(x) for x in c)
        eps = err / acc
        epsf = errf / accf
        rows[f"{p:g}|k{kappa}|lru{int(lru)}|m{depth}"] = {
            "accepted": acc, "errors": err, "eps": eps,
            "eps_se": float(np.sqrt(eps / acc)),
            "accepted_flag": accf, "eps_flag": epsf,
            "eps_flag_se": float(np.sqrt(epsf / accf))}
    out["u1f"] = rows
    print("LRB-D unif=1+final accepted error under leakage (new seeds):")
    for k, v in rows.items():
        print(f"  {k:24s} eps = {v['eps']:.3e} +- {v['eps_se']:.1e} "
              f"(accepted {v['accepted']}); with leakage-flag rejection "
              f"{v['eps_flag']:.3e} +- {v['eps_flag_se']:.1e}")

    # const=1 control: extra Pauli noise instead of leakage (stim, no leakage)
    p = 2e-3
    depths = [0, 2, 4, 6, 10, 14, 18, 20, 22]
    cshots = args.const1_shots
    ctasks = []
    factors = (1.0, 1.25, 1.5)
    for f in factors:
        for si in range(12):
            for depth in depths:
                ctasks.append((p * f, depth, si, cshots,
                               313_001 + 1009 * si + 7 * depth
                               + int(1000 * f)))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        cres = list(pool.map(_const1_task, ctasks))
    F = {f: np.full((12, len(depths)), np.nan) for f in factors}
    for (pp, depth, si, n, seed), acc, err in cres:
        f = round(pp / p, 4)
        F[f][si, depths.index(depth)] = abs(1 - 2 * err / acc) if acc else np.nan
    ctrl = {}
    pairs = [(0, 4), (4, 10), (10, 22)]
    for f in factors[1:]:
        R = np.nanmean(F[f], 0) / np.nanmean(F[1.0], 0)
        rates = [float(x) for x in _local_rates(R, depths, pairs)]
        ctrl[f"x{f}"] = {"R": R.tolist(), "rates": rates,
                         "F": np.nanmean(F[f], 0).tolist()}
    ctrl["F_base"] = np.nanmean(F[1.0], 0).tolist()
    out["const1_control"] = ctrl
    print("const=1 control, F(p*x)/F(p) at p=2e-3 without leakage:")
    print("  F base " + ", ".join(f"{x:.3f}" for x in ctrl["F_base"]))
    for f in factors[1:]:
        c = ctrl[f"x{f}"]
        print(f"  x{f}: R(4,10,22) = {c['R'][2]:.4f}, {c['R'][4]:.4f}, "
              f"{c['R'][8]:.4f}; local rates " + ", ".join(
                  f"{r:.4f}" for r in c["rates"]))
    _dump(Path(args.out), "leak_sim.json", out)
    return out


# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=[
        "capacity", "erasure", "census", "flags", "flagE1", "leakfo",
        "leaktables", "erasuretables", "heraldcal", "leaksim", "all"])
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--num-seq", type=int, default=6)
    parser.add_argument("--depths", default=None)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--shots", type=int, default=None)
    parser.add_argument("--const1-shots", type=int, default=100_000)
    parser.add_argument("--cal-shots", type=int, default=400_000)
    parser.add_argument("--cal-seed", type=int, default=4242)
    args = parser.parse_args()
    start = time.perf_counter()
    commands = {"capacity": cmd_capacity, "erasure": cmd_erasure,
                "census": cmd_census, "flags": cmd_flags,
                "flagE1": cmd_flag_e1, "leakfo": cmd_leakfo,
                "leaktables": cmd_leaktables,
                "erasuretables": cmd_erasuretables,
                "heraldcal": cmd_heraldcal, "leaksim": cmd_leaksim}
    names = ([args.command] if args.command != "all" else
             ["capacity", "flags", "erasure", "census", "leaktables",
              "erasuretables", "heraldcal", "leakfo", "flagE1", "leaksim"])
    for name in names:
        print(f"===== {name} =====", flush=True)
        commands[name](args)
    print(f"done in {time.perf_counter() - start:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
