#!/usr/bin/env python
"""Adversarial verification of work package A (sdim cross-check of the qubit model).

Written independently of ``studies/correction-decoders/qubit/scripts/crosscheck_sdim.py``: nothing is
imported from it, and the qubit code definition, the sdim -> Stim translation,
the detector parsing and the comparisons are all built differently.

Subcommands (outputs go to ``studies/correction-decoders/qubit/results/A-verify/``):

``dem``
    Claim 4. Circuits come from the repo's ``LRBCircuitGenerator`` with a qubit
    ``LRBCodeDefinition`` derived *mechanically* from the production
    ``folded_qutrit`` definition (every qutrit template is rebuilt at d = 2
    with H_INV->H, CNOT_INV->CNOT, CZ_INV->CZ, X_INV->X, Z_INV->Z; the detector
    wires and the qutrit +-1 terminal terms are kept unchanged). They are
    translated to Stim by gate id, with detectors taken from sdim's own parser
    (``Program._build_ir``) and probed with unit vectors. Each circuit is
    compared with ``lrb_qubit.circuits`` in three ways:

    1. per-qubit timelines (exact equality of every gate and noise channel,
       including probabilities, up to reordering of operations on disjoint
       qubits);
    2. the detector error model in sdim's raw-detector basis (lead body plus
       the repo's detectors) and in the lead's difference-detector basis
       (repo body plus the lead's detectors);
    3. the production route: ``write_circuit`` -> ``read_circuit`` ->
       ``ensure_noise_params`` (what the runtime actually simulates).

    New sequences (not the implementer's three), p in {0.01, 0.0037}, depths
    {0, 3, 7} generated in one call, data2 LRB and const=0, plus all 24
    descriptors at depth 1, plus negative controls.
``floor``
    Claim 8. Undetectable-logical probability of the repo-generated data2
    circuits for several p, its small-p limit, a closed form, and the circuit
    locations that produce it.
``semantics``
    Claims 1-2. sdim d = 2 gate unitaries vs Stim, CNOT direction, and a
    random-circuit replay with mid-circuit M and RESET against Stim's
    TableauSimulator (also with S and S_DAG swapped, to show what such a
    test can and cannot see).
``hazards``
    The two latent repo hazards, plus two further reuse hazards found here.
``sample``
    Sampling spot check on 8 of the new circuits at p = 0.01, against exact
    DEM values and against Stim. ``--backend sdim`` uses sdim's public
    ``Program.simulate``; ``--backend repo_dem`` uses the repo's production
    sampler ``lrb.dem_simulation.simulate_circuit_with_dem`` (the
    ``--simulation-backend dem`` path that produced the qutrit production run).

Reported runs (shared allocation, 16 cores)::

    python studies/correction-decoders/qubit/scripts/verify_crosscheck.py sample --workers 16 \
        --chunks 25 --chunk-shots 20000 --stim-shots 4000000          # 137 s
    python studies/correction-decoders/qubit/scripts/verify_crosscheck.py sample --backend repo_dem \
        --workers 16 --chunks 25 --chunk-shots 40000 --stim-shots 4000000  # 77 s

``followup``
    Pre-registered fresh-seed replication of the one extreme cell of the
    ``repo_dem`` run (login node, about a minute).

``dem``, ``floor``, ``semantics`` and ``hazards`` take seconds to a minute
on the login node.

Run from the repository root::

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    export PYTHONPATH=$PWD/src
    python studies/correction-decoders/qubit/scripts/verify_crosscheck.py dem
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import random
import re
import sys
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from fractions import Fraction
from pathlib import Path

import numpy as np
import stim

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from sdim.circuit import Circuit  # noqa: E402
from sdim.circuit_io import read_circuit, write_circuit  # noqa: E402
from sdim.gatedata import GateData  # noqa: E402
from sdim.program import Program  # noqa: E402

from lrb import code_definitions as D  # noqa: E402
from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_simulation_profiles import (  # noqa: E402
    CodeSimulationProfileRegistry, _UnpackSpec)
from lrb.lrb_simulation import LRBSimulationPipeline, NoiseModelUtils  # noqa: E402

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as QC  # noqa: E402
from lrb_qubit import experiment as QE  # noqa: E402

OUT = ROOT / "studies" / "correction-decoders" / "qubit" / "results" / "A-verify"

NEW_SEED = 9_170_331          # not the grid seed 20261028
NEW_DEPTHS = (0, 3, 7)
NEW_PROBS = (0.01, 0.0037)
# Hand-made sequence: every token type (I, H, P, X, Z), a five-token descriptor.
HAND_SEQUENCE = ["XZHPH", "I", "ZPH", "XHP", "PH", "HPH", "XZP"]
SILENT = (frozenset(), frozenset({0}))


# ---------------------------------------------------------------------------
# Qubit code definition derived from the production folded_qutrit definition
# ---------------------------------------------------------------------------

D2_RENAME = {"H_INV": "H", "CNOT_INV": "CNOT", "CZ_INV": "CZ",
             "X_INV": "X", "Z_INV": "Z"}


def to_d2(c3: Circuit, swap_phase: bool = False) -> Circuit:
    """Rebuild a d = 3 template at d = 2 (gates that are self-inverse at d = 2 renamed)."""
    if c3.dimension != 3:
        raise ValueError("expected a qutrit template")
    c2 = Circuit(num_qudits=c3.num_qudits, dimension=2)
    for op in c3.operations:
        name = D2_RENAME.get(op.gate_name, op.gate_name)
        if swap_phase:
            name = {"P": "P_INV", "P_INV": "P"}.get(name, name)
        params = dict(op.params or {})
        if op.target_index is None:
            c2.add_gate(name, op.qudit_index, **params)
        else:
            c2.add_gate(name, op.qudit_index, op.target_index, **params)
    return c2


def qubit_definition(swap_phase: bool = False):
    """``folded_qutrit`` with every circuit callback wrapped by ``to_d2``.

    Everything else (z_ALT/X check blocks, reset block, readouts, detector
    wires, logical terms and the qutrit terminal X terms with -1
    coefficients) is taken unchanged from the production registry. Only the
    Clifford catalog is the qubit one (P = S), with inverses built by the
    repo's own ``SINGLE_QUTRIT_INVERSES`` rule.
    """
    base = CodeSimulationProfileRegistry._build_code_definition("folded_qutrit")
    assert base.dimension == 3

    def wrap(factory):
        return lambda *args: to_d2(factory(*args), swap_phase)

    catalog = tuple(QC.SINGLE_QUBIT_CLIFFORDS)
    return dataclasses.replace(
        base,
        dimension=2,
        clifford_strings=catalog,
        clifford_inverse_map={c: [D.SINGLE_QUTRIT_INVERSES[g] for g in c]
                              for c in catalog},
        apply_physical_gate=lambda circ, g: circ.add_gate(
            D2_RENAME.get(g, g), 0),
        logical_plus_initial_state=wrap(base.logical_plus_initial_state),
        logical_gate_circuit=wrap(base.logical_gate_circuit),
        stabilizer_check_blocks=tuple((wrap(f), wires) for f, wires
                                      in base.stabilizer_check_blocks),
        reset_measurement_wires=wrap(base.reset_measurement_wires),
        terminal_measurement=wrap(base.terminal_measurement),
        terminal_const0_measurement=wrap(base.terminal_const0_measurement),
    )


def repo_circuits(defn, sequence, depths, p, kind, chp_dir=None):
    """Generate one depth family the way ``generate_tests`` does.

    ``clifford_strings`` has exactly ``max(depths)`` entries. Returns
    ``{depth: (in_memory_circuit, chp_round_trip_circuit_or_None)}``. The
    .chp file is written right after ``update_noise_param`` and before
    ``ensure_noise_params``, as in production.
    """
    gen = LRBCircuitGenerator(code_definition=defn, noise_model="si1000")
    depths = sorted(depths)
    strings = list(sequence[:depths[-1]])
    build = (gen.generate_lrb_clifford_sequence if kind == "lrb"
             else gen.generate_lrb_const0_clifford_sequence)
    circuits = build(depths=depths, with_noise=True, clifford_strings=strings)
    out = {}
    for depth, c in zip(depths, circuits):
        gen.update_noise_param(c, p)
        round_trip = None
        if chp_dir is not None:
            path = write_circuit(circuit=c, output_file=f"{kind}_{depth}.chp",
                                 comment="", directory=chp_dir)
            round_trip = read_circuit(os.path.abspath(path))
            NoiseModelUtils.ensure_noise_params(round_trip)
        NoiseModelUtils.ensure_noise_params(c)
        out[depth] = (c, round_trip)
    return out


# ---------------------------------------------------------------------------
# sdim -> Stim translation
# ---------------------------------------------------------------------------

_GD2 = GateData(2)
_NAME_OF_ID = {gate.gate_id: name for name, gate in _GD2.gateMap.items()}
STIM_1Q = {"I": "I", "X": "X", "X_INV": "X", "Z": "Z", "Z_INV": "Z",
           "H": "H", "H_INV": "H", "P": "S", "P_INV": "S_DAG"}
STIM_2Q = {"CNOT": "CX", "CNOT_INV": "CX", "CZ": "CZ", "CZ_INV": "CZ",
           "SWAP": "SWAP"}
_TERM = re.compile(r"([+-]?)\s*(?:(\d+)\s*\*\s*)?rec\[(-?\d+)\]")


def sdim_annotation_forms(circuit: Circuit) -> list[dict]:
    """Linear forms (mod 2) of every DETECTOR / LOGICAL_OBSERVABLE as sdim parses them.

    Uses sdim's own ``_build_ir`` (the code that builds the detector lambdas
    the frame simulator evaluates) and probes each lambda with unit vectors.
    Also parses the expression text directly; ``parse_ok`` says whether the
    two agree (sdim rewrites ``[k]`` placeholders with sequential
    ``str.replace``, which can collide).
    """
    state = np.random.get_state()
    program = Program(circuit)
    _, _, info = program._build_ir(program.circuits, 1)
    np.random.set_state(state)
    ops = [op for op in circuit.operations
           if _NAME_OF_ID[op.gate_id] in ("DETECTOR", "LOGICAL_OBSERVABLE")]
    forms = []
    total = info.total_measurements
    for k, (index, label, arguments, is_logical) in enumerate(
            info.detector_data):
        fn = info.detector_functions[index]
        n = len(arguments)
        base = int(fn([0] * n)) % 2
        support = set()
        for j, a in enumerate(arguments):
            e = [0] * n
            e[j] = 1
            if (int(fn(e)) - base) % 2:
                support ^= {a}
        rng = np.random.default_rng(k)
        for _ in range(8):                      # linearity check
            v = rng.integers(0, 2, n).tolist()
            pred = (base + sum(v[j] for j, a in enumerate(arguments)
                               if a in support)) % 2
            if int(fn(v)) % 2 != pred:
                raise ValueError(f"non-linear detector {label}")
        text = {}
        for sign, coef, idx in _TERM.findall(ops[k].params["expr"]):
            i = int(idx) % total
            text[i] = (text.get(i, 0) + int(coef or 1)) % 2
        intended = {i for i, v in text.items() if v}
        forms.append({"label": label, "logical": bool(is_logical),
                      "constant": base, "support": frozenset(support),
                      "parse_ok": intended == support})
    return forms


def translate(circuit: Circuit) -> tuple[stim.Circuit, dict]:
    """Gate-for-gate Stim translation of a d = 2 sdim circuit.

    Returns the circuit and a small report (parse agreement, the worst
    deviation of an N2 distribution from the uniform expansion of its ``prob``).
    """
    if circuit.dimension != 2:
        raise ValueError("only d = 2")
    forms = sdim_annotation_forms(circuit)
    out = stim.Circuit()
    measurements = 0
    k = 0
    n2_dev = 0.0
    for op in circuit.operations:
        name = _NAME_OF_ID[op.gate_id]
        q, t = op.qudit_index, op.target_index
        if name in STIM_1Q:
            out.append(STIM_1Q[name], [q])
        elif name in STIM_2Q:
            out.append(STIM_2Q[name], [q, t])
        elif name == "M":
            out.append("M", [q])
            measurements += 1
        elif name == "RESET":
            out.append("R", [q])
        elif name == "N1":
            prob = float(op.params["prob"])
            gate = {"d": "DEPOLARIZE1", "f": "X_ERROR",
                    "p": "Z_ERROR"}[op.params["noise_channel"]]
            if prob > 0:
                out.append(gate, [q], prob)
        elif name == "N2":
            prob = float(op.params["prob"])
            # sdim 1.4.0 reads an N2 with only ``prob`` as uniform two-qubit
            # depolarizing; an explicit ``prob_dist`` must be that law.
            if "prob_dist" in op.params:
                dist = np.asarray(op.params["prob_dist"], dtype=float)
                expected = np.asarray(
                    NoiseModelUtils.two_qudit_depol_prob_dist(2, prob))
                if dist.shape != (16,) or not np.array_equal(dist, expected):
                    raise ValueError(
                        "N2 is not the uniform expansion of its prob")
                n2_dev = max(n2_dev, abs(dist[1:].sum() - prob))
            if prob > 0:
                out.append("DEPOLARIZE2", [q, t], prob)
        elif name in ("DETECTOR", "LOGICAL_OBSERVABLE"):
            form = forms[k]
            k += 1
            targets = [stim.target_rec(i - measurements)
                       for i in sorted(form["support"])]
            if form["logical"]:
                out.append("OBSERVABLE_INCLUDE", targets, 0)
            else:
                out.append("DETECTOR", targets)
        elif name == "TICK":
            continue
        else:
            raise ValueError(f"untranslatable sdim gate {name}")
    report = {"parse_ok": all(f["parse_ok"] for f in forms),
              "constant_terms": sum(f["constant"] for f in forms),
              "n2_max_tail_minus_prob": n2_dev}
    return out, report


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------

ANNOTATIONS = ("DETECTOR", "OBSERVABLE_INCLUDE")
TWO_QUBIT = {"CX", "CZ", "SWAP", "DEPOLARIZE2"}
SYMMETRIC = {"CZ", "SWAP", "DEPOLARIZE2"}


def split(c: stim.Circuit) -> tuple[stim.Circuit, stim.Circuit]:
    """(body, annotations); annotations must all follow the last measurement."""
    body, ann = stim.Circuit(), stim.Circuit()
    for inst in c.flattened():
        if inst.name in ANNOTATIONS:
            ann.append(inst)
        else:
            if len(ann) and inst.name == "M":
                raise ValueError("measurement after an annotation")
            body.append(inst)
    return body, ann


def measured_qubits(c: stim.Circuit) -> list[int]:
    return [t.value for inst in c.flattened() if inst.name == "M"
            for t in inst.targets_copy()]


def timelines(c: stim.Circuit) -> dict:
    """Per-qubit sequence of (gate, targets, args) letters, annotations dropped.

    Two circuits with equal timelines are equal up to reordering operations
    that act on disjoint qubits (Mazurkiewicz trace equivalence).
    """
    lines = defaultdict(list)
    for inst in c.flattened():
        if inst.name in ANNOTATIONS or inst.name == "TICK":
            continue
        args = tuple(inst.gate_args_copy())
        ts = [t.value for t in inst.targets_copy()]
        width = 2 if inst.name in TWO_QUBIT else 1
        for i in range(0, len(ts), width):
            group = tuple(ts[i:i + width])
            if inst.name in SYMMETRIC:
                group = tuple(sorted(group))
            for q in group:
                lines[q].append((inst.name, group, args))
    return dict(lines)


def first_timeline_difference(a: dict, b: dict) -> str:
    for q in sorted(set(a) | set(b)):
        la, lb = a.get(q, []), b.get(q, [])
        for i, (x, y) in enumerate(zip(la, lb)):
            if x != y:
                return f"q{q}[{i}]: {x} vs {y}"
        if len(la) != len(lb):
            return f"q{q}: length {len(la)} vs {len(lb)}"
    return ""


def mechanisms(c: stim.Circuit) -> dict:
    """{(detector set, observable set): probability}, equal symptoms XOR-merged."""
    out: dict = {}
    for inst in c.detector_error_model().flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        ts = inst.targets_copy()
        key = (frozenset(t.val for t in ts if t.is_relative_detector_id()),
               frozenset(t.val for t in ts if t.is_logical_observable_id()))
        old = out.get(key)
        out[key] = p if old is None else old * (1 - p) + p * (1 - old)
    return out


def compare_mechanisms(a: dict, b: dict) -> dict:
    keys = set(a) | set(b)
    rel = 0.0
    for k in keys:
        x, y = a.get(k, 0.0), b.get(k, 0.0)
        if x != y:
            rel = max(rel, abs(x - y) / max(x, y))
    return {"same_symptoms": set(a) == set(b),
            "bit_identical": set(a) == set(b) and all(a[k] == b[k] for k in a),
            "max_rel": rel, "n_a": len(a), "n_b": len(b)}


def lead_circuit(kind, sequence, depth, p, noise=None, terminal="data2"):
    noise = Q.SI1000() if noise is None else noise
    if kind == "lrb":
        return Q.build_lrb_circuit(sequence, depth, p, terminal=terminal,
                                   noise=noise).circuit
    return Q.build_lrb_const0_circuit(sequence, depth, p, noise=noise).circuit


def compare(translated: stim.Circuit, lead: stim.Circuit) -> dict:
    """Timeline and two-basis DEM comparison of two Stim circuits."""
    row = {}
    row["meas_seq_equal"] = measured_qubits(translated) == measured_qubits(lead)
    ta, tb = timelines(translated), timelines(lead)
    row["timeline_equal"] = ta == tb
    row["timeline_first_diff"] = first_timeline_difference(ta, tb)
    m_repo = mechanisms(translated)
    m_lead = mechanisms(lead)
    row["silent_logical_repo"] = m_repo.get(SILENT, 0.0)
    row["silent_logical_lead"] = m_lead.get(SILENT, 0.0)
    if row["meas_seq_equal"]:
        body_r, ann_r = split(translated)
        body_l, ann_l = split(lead)
        repo_basis = compare_mechanisms(m_repo, mechanisms(body_l + ann_r))
        lead_basis = compare_mechanisms(m_lead, mechanisms(body_r + ann_l))
    else:
        repo_basis = lead_basis = {"same_symptoms": False,
                                   "bit_identical": False,
                                   "max_rel": float("nan"),
                                   "n_a": len(m_repo), "n_b": len(m_lead)}
    for prefix, res in (("repo_basis", repo_basis), ("lead_basis", lead_basis)):
        for key, value in res.items():
            row[f"{prefix}_{key}"] = value
    row["dem_identical"] = (repo_basis["bit_identical"]
                            and lead_basis["bit_identical"])
    row["num_detectors"] = translated.num_detectors
    row["num_detectors_lead"] = lead.num_detectors
    return row


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list = []
    for r in rows:
        keys += [k for k in r if k not in keys]

    def fmt(v):
        if isinstance(v, float):
            return f"{v:.17g}"
        return v
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt(r.get(k, "")) for k in keys})


def new_sequences() -> dict[str, list[str]]:
    rng = np.random.default_rng(NEW_SEED)
    seqs = {"new_A": Q.random_clifford_sequence(max(NEW_DEPTHS), rng),
            "new_B": Q.random_clifford_sequence(max(NEW_DEPTHS), rng),
            "hand_C": list(HAND_SEQUENCE)}
    implementer = [s[:max(NEW_DEPTHS)]
                   for s in QE.clifford_sequences(3, 22, 20261028)]
    for name, s in seqs.items():
        if s in implementer:
            raise RuntimeError(f"{name} coincides with an implementer sequence")
    return seqs


# ---------------------------------------------------------------------------
# dem: claim 4
# ---------------------------------------------------------------------------

def cmd_dem(args) -> None:
    out = Path(args.out)
    defn = qubit_definition()
    seqs = new_sequences()
    rows = []
    t0 = time.time()
    with tempfile.TemporaryDirectory() as tmp:
        for name, seq in seqs.items():
            for p in NEW_PROBS:
                for kind in ("lrb", "const0"):
                    fam = repo_circuits(defn, seq, NEW_DEPTHS, p, kind,
                                        chp_dir=os.path.join(tmp, f"{name}_{p}"))
                    for depth, (c_mem, c_chp) in fam.items():
                        tr, rep = translate(c_mem)
                        tr_chp, rep_chp = translate(c_chp)
                        lead = lead_circuit(kind, seq, depth, p)
                        row = {"set": "main", "sequence": name,
                               "descriptors": " ".join(seq[:depth]),
                               "kind": kind, "depth": depth, "p": p}
                        row.update(compare(tr, lead))
                        row.update({f"translate_{k}": v for k, v in rep.items()})
                        row["chp_route_timeline_equal_inmem"] = (
                            timelines(tr_chp) == timelines(tr))
                        row["chp_route_vs_lead_dem_identical"] = compare(
                            tr_chp, lead)["dem_identical"]
                        row["chp_parse_ok"] = rep_chp["parse_ok"]
                        rows.append(row)
                        print(f"{name} p={p} {kind:6s} m={depth}: "
                              f"timeline={row['timeline_equal']} "
                              f"dem={row['dem_identical']} "
                              f"(repo-basis rel {row['repo_basis_max_rel']:.1e},"
                              f" lead-basis rel {row['lead_basis_max_rel']:.1e},"
                              f" mech {row['repo_basis_n_a']}) "
                              f"chp={row['chp_route_vs_lead_dem_identical']} "
                              f"silent/p={row['silent_logical_repo'] / p:.6f}",
                              flush=True)
        # Every descriptor at depth 1.
        for descriptor in QC.SINGLE_QUBIT_CLIFFORDS:
            for kind in ("lrb", "const0"):
                fam = repo_circuits(defn, [descriptor], [1], 0.01, kind)
                tr, rep = translate(fam[1][0])
                lead = lead_circuit(kind, [descriptor], 1, 0.01)
                row = {"set": "descriptor", "sequence": descriptor,
                       "descriptors": descriptor, "kind": kind, "depth": 1,
                       "p": 0.01}
                row.update(compare(tr, lead))
                row.update({f"translate_{k}": v for k, v in rep.items()})
                rows.append(row)
    write_csv(out / "dem_new_sequences.csv", rows)
    main = [r for r in rows if r["set"] == "main"]
    desc = [r for r in rows if r["set"] == "descriptor"]
    for label, group in (("main", main), ("descriptor", desc)):
        print(f"[{label}] {len(group)} circuits: "
              f"timeline equal {sum(r['timeline_equal'] for r in group)}, "
              f"DEM bit-identical (both bases) "
              f"{sum(r['dem_identical'] for r in group)}, "
              f"max rel {max(max(r['repo_basis_max_rel'], r['lead_basis_max_rel']) for r in group):.1e}, "
              f"sdim parse ok {sum(r['translate_parse_ok'] for r in group)}")
    print(f"[chp route] timeline equal to in-memory: "
          f"{sum(r['chp_route_timeline_equal_inmem'] for r in main)}/{len(main)}, "
          f"DEM identical to lead: "
          f"{sum(r['chp_route_vs_lead_dem_identical'] for r in main)}/{len(main)}")

    # Negative controls (sequence new_A, depth 7, p = 0.01).
    seq = seqs["new_A"]
    p, depth = 0.01, 7
    changed = list(seq)
    changed[2] = "H" if seq[2] != "H" else "P"
    controls = []

    def run_control(name, kind, translated, lead, expect_flag=True):
        row = {"control": name, "kind": kind, "depth": depth, "p": p,
               "expected": "flag" if expect_flag else "blind spot (identical DEM)"}
        row.update(compare(translated, lead))
        row["flagged_by_dem"] = not row["dem_identical"]
        row["flagged_by_timeline"] = not row["timeline_equal"]
        row["flagged_by_meas_seq"] = not row["meas_seq_equal"]
        row["flagged"] = (row["flagged_by_dem"] or row["flagged_by_timeline"]
                          or row["flagged_by_meas_seq"])
        controls.append(row)
        print(f"control {name:34s} {kind:6s}: flagged={row['flagged']} "
              f"(dem {row['flagged_by_dem']}, timeline "
              f"{row['flagged_by_timeline']}, meas {row['flagged_by_meas_seq']}) "
              f"rel repo/lead basis {row['repo_basis_max_rel']:.2e}/"
              f"{row['lead_basis_max_rel']:.2e}")

    base_tr = {kind: translate(repo_circuits(defn, seq, NEW_DEPTHS, p, kind)
                               [depth][0])[0] for kind in ("lrb", "const0")}
    for name, kind, noise in (
            ("lead meas_flip x1.001", "lrb", Q.SI1000().scaled(meas_flip=1.001)),
            ("lead reset_flip x0.999", "lrb", Q.SI1000().scaled(reset_flip=0.999)),
            ("lead two_qubit x1.001", "lrb", Q.SI1000().scaled(two_qubit=1.001)),
            ("lead meas_depol x1.001", "lrb", Q.SI1000().scaled(meas_depol=1.001)),
            ("lead two_qubit x1.001", "const0", Q.SI1000().scaled(two_qubit=1.001)),
            ("lead one_qubit x1.001", "const0", Q.SI1000().scaled(one_qubit=1.001))):
        run_control(name, kind, base_tr[kind],
                    lead_circuit(kind, seq, depth, p, noise=noise))
    run_control("lead terminal=data5", "lrb", base_tr["lrb"],
                lead_circuit("lrb", seq, depth, p, terminal="data5"))
    for kind in ("lrb", "const0"):
        run_control(f"lead descriptor 3 -> {changed[2]}", kind, base_tr[kind],
                    lead_circuit(kind, changed, depth, p))
    # Repo-side perturbation: first 'f' channel at 5p multiplied by 1.001.
    c = repo_circuits(defn, seq, NEW_DEPTHS, p, "lrb")[depth][0]
    for op in c.operations:
        if op.gate_name == "N1" and op.params.get("noise_channel") == "f" \
                and abs(float(op.params["prob"]) - 5 * p) < 1e-15:
            op.params["prob"] = float(op.params["prob"]) * 1.001
            break
    run_control("repo first meas X_ERROR x1.001", "lrb", translate(c)[0],
                lead_circuit("lrb", seq, depth, p))
    # Known blind spot: S <-> S_DAG everywhere leaves every DEM unchanged.
    swapped = qubit_definition(swap_phase=True)
    for kind in ("lrb", "const0"):
        c = repo_circuits(swapped, seq, NEW_DEPTHS, p, kind)[depth][0]
        run_control("repo P<->P_INV swapped", kind, translate(c)[0],
                    lead_circuit(kind, seq, depth, p), expect_flag=False)
    write_csv(out / "dem_negative_controls.csv", controls)
    print(f"done in {time.time() - t0:.1f} s")


# ---------------------------------------------------------------------------
# floor: claim 8
# ---------------------------------------------------------------------------

def floor_closed_form(p: float) -> float:
    """Silent-logical DEM mass implied by the 68/15 decomposition.

    Three DEPOLARIZE1(2p) idles (data 0 at the M7 and M8 layers, data 3 at
    the M8 layer) each give 4p/3 exactly after Stim's merge, and the two
    DEPOLARIZE2(p) after CX 7->0 and CX 8->3 give 4 of 15 independent
    components each, so prod(1 - 2 q_i) = (1 - 8p/3)^3 (1 - 16p/15).
    """
    return (1 - (1 - 8 * p / 3) ** 3 * (1 - 16 * p / 15)) / 2


def cmd_floor(args) -> None:
    out = Path(args.out)
    defn = qubit_definition()
    seqs = new_sequences()
    rows = []
    for p in (1e-7, 1e-4, 2e-3, 0.005, 0.01, 0.02):
        values = []
        for name, seq in seqs.items():
            fam = repo_circuits(defn, seq, NEW_DEPTHS, p, "lrb")
            for depth, (c, _) in fam.items():
                tr, _ = translate(c)
                repo = mechanisms(tr).get(SILENT, 0.0)
                lead = mechanisms(lead_circuit("lrb", seq, depth, p)).get(SILENT, 0.0)
                values.append((name, depth, repo, lead))
        repo_vals = {v[2] for v in values}
        lead_vals = {v[3] for v in values}
        row = {"p": p, "circuits": len(values),
               "distinct_repo_values": len(repo_vals),
               "distinct_lead_values": len(lead_vals),
               "repo_equals_lead": repo_vals == lead_vals,
               "silent_over_p": min(repo_vals) / p,
               "closed_form_over_p": floor_closed_form(p) / p,
               "closed_form_rel_err": abs(min(repo_vals) - floor_closed_form(p))
               / floor_closed_form(p),
               "68/15": 68 / 15}
        rows.append(row)
        print(row, flush=True)
    write_csv(out / "floor.csv", rows)

    # Where the floor comes from (first order, exact fractions of p).
    p = 1e-3
    seq = seqs["new_A"]
    c = repo_circuits(defn, seq, NEW_DEPTHS, p, "lrb")[3][0]
    tr, _ = translate(c)
    flat = tr.flattened()
    explained = tr.explain_detector_error_model_errors(
        dem_filter=stim.DetectorErrorModel("error(1) L0"),
        reduce_to_one_representative_error=False)
    loc_rows = []
    total = Fraction(0)
    for e in explained:
        for loc in e.circuit_error_locations:
            gate = loc.instruction_targets.gate
            arg = loc.instruction_targets.args[0]
            scale = Fraction(round(arg / p * 10), 10)
            weight = scale / {"DEPOLARIZE1": 3, "DEPOLARIZE2": 15}.get(gate, 1)
            offset = loc.stack_frames[0].instruction_offset
            before = next((f"{flat[j].name} {[t.value for t in flat[j].targets_copy()]}"
                           for j in range(offset - 1, -1, -1)
                           if flat[j].name not in ("DEPOLARIZE1", "DEPOLARIZE2",
                                                   "X_ERROR")), "")
            nxt = next((f"M {[t.value for t in flat[j].targets_copy()]}"
                        for j in range(offset + 1, len(flat))
                        if flat[j].name == "M"), "")
            pauli = "*".join(f"{t.gate_target.pauli_type}{t.gate_target.value}"
                             for t in loc.flipped_pauli_product)
            total += weight
            loc_rows.append({"gate": gate, "arg_over_p": float(scale),
                             "pauli": pauli, "weight_over_p": str(weight),
                             "after_gate": before, "next_measurement": nxt,
                             "instruction_offset": offset})
    write_csv(out / "floor_locations.csv", loc_rows)
    print(f"first-order floor from {len(loc_rows)} circuit error components: "
          f"{total} p = {float(total):.6f} p (68/15 = {68 / 15:.6f})")
    for r in loc_rows:
        print("  ", r)


# ---------------------------------------------------------------------------
# semantics: claims 1-2
# ---------------------------------------------------------------------------

def _sdim_unitary(name: str, qudits: tuple[int, ...]) -> np.ndarray:
    import cirq
    from sdim.circuit_io import circuit_to_cirq_circuit
    c = Circuit(num_qudits=len(qudits), dimension=2)
    c.add_gate(name, *qudits)
    return cirq.unitary(circuit_to_cirq_circuit(c))


def _equal_up_to_phase(a: np.ndarray, b: np.ndarray) -> bool:
    idx = np.unravel_index(np.argmax(np.abs(b)), b.shape)
    if abs(a[idx]) < 1e-12:
        return False
    return np.allclose(a * (b[idx] / a[idx]), b, atol=1e-12)


def _replay(n: int, ops: list, phase_map: dict) -> tuple[int, int, int]:
    """Run ``ops`` in sdim (one tableau shot) and replay them in Stim.

    Stim post-selects every random outcome to sdim's value. Returns
    (deterministic agreements, random agreements, mismatches).
    """
    c = Circuit(num_qudits=n, dimension=2)
    for name, qs in ops:
        c.add_gate(name, *qs)
    program = Program(c)
    program.simulate(shots=1)
    results = program.measurement_results         # [qudit][slot][shot]
    slot = [0] * n
    sim = stim.TableauSimulator()
    sim.set_num_qubits(n)
    det = rnd = bad = 0
    stim_names = dict(STIM_1Q, **phase_map)
    for name, qs in ops:
        if name in ("M", "RESET"):
            q = qs[0]
            r = results[q][slot[q]][0]
            slot[q] += 1
            z = sim.peek_z(q)
            if z != 0:
                ok = r.deterministic and int(r.measurement_value) == (z == -1)
                det += ok
                bad += not ok
            else:
                ok = not r.deterministic
                rnd += ok
                bad += not ok
                sim.postselect_z(q, desired_value=bool(r.measurement_value))
            if name == "RESET":
                sim.reset(q)
            else:
                sim.measure(q)
        elif name in STIM_2Q:
            sim.do(stim.Circuit(f"{STIM_2Q[name]} {qs[0]} {qs[1]}"))
        else:
            sim.do(stim.Circuit(f"{stim_names[name]} {qs[0]}"))
    return det, rnd, bad


def cmd_semantics(args) -> None:
    out = Path(args.out)
    rows = []
    # (a) Unitaries of sdim's d = 2 gates (cirq export) vs Stim.
    for name, stim_name, qs in (
            ("H", "H", (0,)), ("H_INV", "H", (0,)), ("P", "S", (0,)),
            ("P_INV", "S_DAG", (0,)), ("X", "X", (0,)), ("X_INV", "X", (0,)),
            ("Z", "Z", (0,)), ("Z_INV", "Z", (0,)), ("CNOT", "CX", (0, 1)),
            ("CNOT_INV", "CX", (0, 1)), ("CZ", "CZ", (0, 1)),
            ("CZ_INV", "CZ", (0, 1))):
        u_sdim = _sdim_unitary(name, qs)
        u_stim = stim.Tableau.from_named_gate(stim_name).to_unitary_matrix(
            endian="big")
        other = {"S": "S_DAG", "S_DAG": "S"}.get(stim_name)
        rows.append({"check": f"unitary {name} == stim {stim_name}",
                     "result": _equal_up_to_phase(u_sdim, u_stim),
                     "detail": ("also equals " + other) if other and
                     _equal_up_to_phase(u_sdim, stim.Tableau.from_named_gate(
                         other).to_unitary_matrix(endian="big")) else ""})
    # (b) CNOT direction in sdim's tableau simulator.
    for prep, expect in ((0, (1, 1)), (1, (0, 1))):
        c = Circuit(num_qudits=2, dimension=2)
        c.add_gate("X", prep)
        c.add_gate("CNOT", 0, 1)
        c.add_gate("M", 0)
        c.add_gate("M", 1)
        prog = Program(c)
        prog.simulate(shots=1)
        got = tuple(int(prog.measurement_results[q][0][0].measurement_value)
                    for q in (0, 1))
        rows.append({"check": f"X{prep}; CNOT 0 1; M 0 1 -> {expect}",
                     "result": got == expect, "detail": f"got {got}"})
    # (c) Random circuits with mid-circuit measurement and reset.
    rng = random.Random(args.seed)
    one = ["H", "H_INV", "P", "P_INV", "X", "X_INV", "Z", "Z_INV"]
    two = ["CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"]
    circuits = []
    for _ in range(args.trials):
        n = rng.randint(2, 5)
        ops = []
        for _ in range(40):
            u = rng.random()
            if u < 0.45:
                ops.append((rng.choice(one), (rng.randrange(n),)))
            elif u < 0.80:
                a, b = rng.sample(range(n), 2)
                ops.append((rng.choice(two), (a, b)))
            elif u < 0.93:
                ops.append(("M", (rng.randrange(n),)))
            else:
                ops.append(("RESET", (rng.randrange(n),)))
        ops += [("M", (q,)) for q in range(n)]
        circuits.append((n, ops))
    for label, phase_map in (("P->S, P_INV->S_DAG", {}),
                             ("P->S_DAG, P_INV->S (conjugated)",
                              {"P": "S_DAG", "P_INV": "S"})):
        det = rnd = bad = 0
        for n, ops in circuits:
            d, r, b = _replay(n, ops, phase_map)
            det, rnd, bad = det + d, rnd + r, bad + b
        rows.append({"check": f"random replay vs Stim ({label})",
                     "result": bad == 0,
                     "detail": f"{args.trials} circuits, {det} deterministic "
                               f"and {rnd} random outcomes agree, {bad} "
                               "mismatches"})
    for r in rows:
        print(r)
    write_csv(out / "semantics.csv", rows)


# ---------------------------------------------------------------------------
# hazards
# ---------------------------------------------------------------------------

def _deterministic(stim_circuit: stim.Circuit) -> bool:
    try:
        stim_circuit.detector_error_model()
        return True
    except ValueError:
        return False


def cmd_hazards(args) -> None:
    out = Path(args.out)
    defn = qubit_definition()
    seq = new_sequences()["new_A"]
    rows = []
    gen = LRBCircuitGenerator(code_definition=defn, noise_model="si1000")

    # (a) Pre-sampled list longer than max(depths).
    for kind, build in (("lrb", gen.generate_lrb_clifford_sequence),
                        ("const0", gen.generate_lrb_const0_clifford_sequence)):
        for strings, label in ((seq[:3], "exactly max(depths)=3"),
                               (seq[:7], "7 descriptors for depths=[3]")):
            c = build(depths=[3], with_noise=True, clifford_strings=list(strings))[0]
            gen.update_noise_param(c, 0.0)
            NoiseModelUtils.ensure_noise_params(c)
            tr, _ = translate(c)
            swaps = sum(1 for op in c.operations if op.gate_name == "SWAP")
            expected_swaps = 2 * sum(d.count("H") for d in seq[:3])
            rows.append({"hazard": "presampled list longer than max(depths)",
                         "kind": kind, "case": label,
                         "swap_gates": swaps,
                         "swap_gates_expected": expected_swaps,
                         "noiseless_detectors_and_logical_deterministic":
                             _deterministic(tr)})

    # (b) Unpack-spec dimension default (3) at d = 2.
    spec2 = _UnpackSpec(stabilizer_wires=(5, 6, 7, 8),
                        logical_measurement_wires=(0, 3),
                        logical_outcome_fn=lambda m, _: (m[0] + m[1]) % 2,
                        terminal_x_measurement_wires=(0, 1, 2, 3, 4),
                        x_stabilizer_check_fn=lambda x: True,
                        logical_observable_terms=defn.logical_observable_terms,
                        terminal_x_stabilizer_terms=defn.terminal_x_stabilizer_terms,
                        dimension=2)
    spec_default = dataclasses.replace(spec2, dimension=_UnpackSpec.__dataclass_fields__["dimension"].default)
    c = repo_circuits(defn, seq, [0], 0.0, "lrb")[0][0]
    seen = {}
    for s in range(200):
        np.random.seed(s)
        random.seed(s)
        output = Program(c).simulate(shots=33)
        ref = output[0]
        key = (int(ref[0][0][0].measurement_value),
               int(ref[3][0][0].measurement_value))
        if key in seen:
            continue
        vals = []
        for spec in (spec2, spec_default):
            _, logical = LRBSimulationPipeline.unpack_measurement_results_from_spec(
                output, 0, 33, spec)
            vals.append(sorted(set(np.asarray(logical).tolist())))
        seen[key] = vals
        if len(seen) == 2:
            break
    for key, (v2, v3) in sorted(seen.items()):
        rows.append({"hazard": "unpack spec dimension default",
                     "kind": "lrb", "case": f"reference (x0, x3) = {key}",
                     "logical_values_dimension_2": v2,
                     "logical_values_default_dimension":
                         f"{v3} (default = {spec_default.dimension})"})

    # (c) Reuse hazard: sdim gate aliases could bypass the SI1000 insertion,
    # which matches on the alias-sensitive ``gate_name``. Under sdim 1.4.0 the
    # generator's moment scheduler rejects a gate name it does not know, so an
    # alias now raises instead of going through silently.
    for alias, canonical in (("S", "P"), ("MZ", "M"), ("CX", "CNOT"),
                             ("MR", "RESET")):
        counts = {}
        for name in (canonical, alias):
            src = Circuit(num_qudits=2, dimension=2)
            if canonical == "CNOT":
                src.add_gate(name, 0, 1)
            else:
                src.add_gate(name, 0)
            try:
                noisy = gen.apply_circuit_level_noise(
                    src, noisy_measurements=True)
            except ValueError as error:
                counts[name] = f"rejected: {error}"
                continue
            counts[name] = sum(op.gate_name in ("N1", "N2") for op in noisy.operations)
        rows.append({"hazard": "sdim alias bypasses SI1000 insertion",
                     "kind": "-", "case": f"{alias} (alias of {canonical})",
                     "noise_ops_canonical": counts[canonical],
                     "noise_ops_alias": counts[alias]})
    # sdim's alias table also maps "R" to the Hadamard, not to reset.
    rows.append({"hazard": "sdim alias R",
                 "kind": "-", "case": "GateData(2).aliasMap['R']",
                 "detail": _GD2.aliasMap.get("R")})

    # (d) sdim's detector parser: sequential str.replace of [k] placeholders.
    c = Circuit(num_qudits=2, dimension=2)
    c.add_gate("M", 0)
    c.add_gate("M", 1)
    for expr in ("rec[0]+rec[1]", "rec[1]+rec[0]"):
        c.add_gate("DETECTOR", expr=expr, label=expr)
    forms = sdim_annotation_forms(c)
    for f in forms:
        rows.append({"hazard": "sdim DETECTOR placeholder collision",
                     "kind": "-", "case": f["label"],
                     "detail": f"sdim evaluates support {sorted(f['support'])}"
                               f" (intended {{0, 1}}); parse_ok={f['parse_ok']}"})
    for r in rows:
        print(r)
    write_csv(out / "hazards.csv", rows)


# ---------------------------------------------------------------------------
# sample: optional spot check of sdim's public sampler
# ---------------------------------------------------------------------------

def _exact_joint(mech: dict, bits: list[tuple[str, int]]) -> np.ndarray:
    size = 1 << len(bits)
    states = np.arange(size)
    dist = np.zeros(size)
    dist[0] = 1.0
    for (dets, obs), p in mech.items():
        mask = 0
        for j, (typ, i) in enumerate(bits):
            if (i in dets) if typ == "d" else (i in obs):
                mask |= 1 << j
        if mask:
            dist = (1 - p) * dist + p * dist[states ^ mask]
    return dist


def _sample_task(task):
    kind, seq, depth, p, shots, seed, backend = task
    defn = qubit_definition()
    c = repo_circuits(defn, seq, NEW_DEPTHS, p, kind)[depth][0]
    np.random.seed(seed % 2**32)
    random.seed(seed)
    if backend == "sdim":
        _, payload = Program(c).simulate(shots=shots + 1)
    else:   # the repo's production sampler (--simulation-backend dem)
        from lrb import dem_simulation
        dem_simulation._MODEL_CACHE.clear()
        _, payload = dem_simulation.simulate_circuit_with_dem(c, shots + 1)
    dets = np.array([d["data"] for d in payload["detectors"]], dtype=np.int64).T % 2
    logical = np.asarray(payload["logicals"][0]["data"], dtype=np.int64) % 2
    return task, dets.astype(bool), logical.astype(bool)


def _summaries(kind: str, depth: int, dets: np.ndarray, logical: np.ndarray) -> dict:
    s = {"n": len(logical), "flip_all": int(logical.sum())}
    for i in range(dets.shape[1]):
        s[f"det{i}"] = int(dets[:, i].sum())
    last = dets[:, -4:] if kind == "lrb" else dets
    acc = ~last.any(axis=1)
    s["acc_terminal"] = int(acc.sum())
    s["flip_acc_terminal"] = int((logical & acc).sum())
    if kind == "lrb":
        acc_all = ~dets.any(axis=1)
        s["acc_all"] = int(acc_all.sum())
        s["flip_acc_all"] = int((logical & acc_all).sum())
    pattern = (last.astype(np.int64) * (1 << np.arange(last.shape[1]))).sum(axis=1)
    pattern = pattern + (logical.astype(np.int64) << last.shape[1])
    s["hist"] = np.bincount(pattern, minlength=1 << (last.shape[1] + 1))
    return s


def cmd_sample(args) -> None:
    from scipy import stats
    out = Path(args.out)
    seqs = new_sequences()
    p = 0.01
    circuits = [(kind, name, depth) for name in ("new_A", "new_B")
                for kind in ("lrb", "const0") for depth in (3, 7)]
    tasks = []
    namespace = 7 if args.backend == "sdim" else 8
    for ci, (kind, name, depth) in enumerate(circuits):
        for chunk in range(args.chunks):
            seed = int(np.random.SeedSequence(
                [namespace, ci, chunk]).generate_state(1)[0])
            tasks.append((kind, tuple(seqs[name]), depth, p, args.chunk_shots,
                          seed, args.backend))
    totals = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task, dets, logical in pool.map(_sample_task, tasks):
            kind, seq, depth = task[0], task[1], task[2]
            key = (kind, seq, depth)
            s = _summaries(kind, depth, dets, logical)
            if key not in totals:
                totals[key] = s
            else:
                for k, v in s.items():
                    totals[key][k] = totals[key][k] + v
    print(f"{args.backend} sampling done in {time.time() - t0:.0f} s", flush=True)

    rows, gof = [], []
    for (kind, seq, depth), s in totals.items():
        name = next(k for k, v in seqs.items() if tuple(v) == seq)
        tr, _ = translate(repo_circuits(qubit_definition(), list(seq),
                                        NEW_DEPTHS, p, kind)[depth][0])
        mech = mechanisms(tr)
        nd = tr.num_detectors
        n = s["n"]
        # Stim reference sample of the lead circuit with the repo's detectors.
        body_l, _ = split(lead_circuit(kind, list(seq), depth, p))
        _, ann_r = split(tr)
        sampler = (body_l + ann_r).compile_detector_sampler(seed=12345 + depth)
        sd, so = sampler.sample(args.stim_shots, separate_observables=True)
        st = _summaries(kind, depth, sd, so[:, 0])
        exact = {"flip_all": _exact_joint(mech, [("o", 0)])[1]}
        for i in range(nd):
            exact[f"det{i}"] = _exact_joint(mech, [("d", i)])[1]
        last = list(range(nd - 4, nd)) if kind == "lrb" else list(range(nd))
        joint = _exact_joint(mech, [("d", i) for i in last] + [("o", 0)])
        exact["acc_terminal"] = joint[0] + joint[1 << len(last)]
        exact["flip_acc_terminal"] = joint[1 << len(last)]
        if kind == "lrb" and nd <= 16:
            full = _exact_joint(mech, [("d", i) for i in range(nd)] + [("o", 0)])
            exact["acc_all"] = full[0] + full[1 << nd]
            exact["flip_acc_all"] = full[1 << nd]
        for q in [k for k in s if k not in ("n", "hist")]:
            k1, k2 = s[q], st[q]
            n1, n2 = n, st["n"]
            row = {"kind": kind, "sequence": name, "depth": depth, "p": p,
                   "quantity": q, "sdim_k": k1, "sdim_n": n1,
                   "stim_k": k2, "stim_n": n2}
            pool_p = (k1 + k2) / (n1 + n2)
            se = math.sqrt(pool_p * (1 - pool_p) * (1 / n1 + 1 / n2))
            row["z_sdim_vs_stim"] = (k1 / n1 - k2 / n2) / se if se > 0 else 0.0
            if q in exact:
                e = exact[q]
                se_e = math.sqrt(e * (1 - e) / n1)
                row["exact"] = e
                row["z_sdim_vs_exact"] = (k1 / n1 - e) / se_e if se_e > 0 else 0.0
            rows.append(row)
        # Goodness of fit of the terminal joint law (pooling cells < 5).
        expected = joint * n
        observed = s["hist"]
        order = np.argsort(expected)
        obs_p, exp_p, acc_o, acc_e = [], [], 0.0, 0.0
        for i in order:
            acc_o += observed[i]
            acc_e += expected[i]
            if acc_e >= 5:
                obs_p.append(acc_o)
                exp_p.append(acc_e)
                acc_o = acc_e = 0.0
        if acc_e > 0 and exp_p:
            obs_p[-1] += acc_o
            exp_p[-1] += acc_e
        chi2 = float(sum((o - e) ** 2 / e for o, e in zip(obs_p, exp_p)))
        dof = len(exp_p) - 1
        gof.append({"kind": kind, "sequence": name, "depth": depth, "n": n,
                    "cells": len(exp_p), "chi2": chi2, "dof": dof,
                    "p_value": float(stats.chi2.sf(chi2, dof))})
    suffix = "" if args.backend == "sdim" else f"_{args.backend}"
    write_csv(out / f"sample_tests{suffix}.csv", rows)
    write_csv(out / f"sample_gof{suffix}.csv", gof)
    zs = np.array([r["z_sdim_vs_exact"] for r in rows if "z_sdim_vs_exact" in r])
    z2 = np.array([r["z_sdim_vs_stim"] for r in rows])
    pv = 2 * stats.norm.sf(np.abs(zs))
    holm = np.minimum(1, np.maximum.accumulate(np.sort(pv) * (len(pv) - np.arange(len(pv)))))
    # (CSV columns are named sdim_* for both backends: "the repo-side sampler".)
    print(f"{args.backend} vs exact: {len(zs)} tests, mean z {zs.mean():+.3f}, "
          f"sd {zs.std():.3f}, max|z| {np.abs(zs).max():.2f}, "
          f"min Holm p {holm.min():.3f}")
    print(f"{args.backend} vs Stim: {len(z2)} tests, mean z {z2.mean():+.3f}, "
          f"sd {z2.std():.3f}, max|z| {np.abs(z2).max():.2f}")
    tot = sum(g["chi2"] for g in gof)
    dof = sum(g["dof"] for g in gof)
    print(f"terminal joint law GOF: chi2 {tot:.1f} on {dof} dof, "
          f"p = {stats.chi2.sf(tot, dof):.3f}")
    for g in gof:
        print(g)


def cmd_followup(args) -> None:
    """Pre-registered replication of the one extreme cell of ``sample --backend repo_dem``.

    Fixed before sampling: det0 (X parity x0+x1+x2) of the const=0 circuit of
    new_A at m = 7, p = 0.01 (z = -3.39 vs exact, Holm p 0.091 in the main
    run), fresh seeds (namespace 9), 10 chunks of 4e5 shots through the
    repo's DEM backend; one two-sided z test at alpha 0.05, plus the
    8-cell joint-law goodness of fit of that circuit.
    """
    from scipy import stats
    from lrb import dem_simulation
    seq = new_sequences()["new_A"]
    c = repo_circuits(qubit_definition(), seq, NEW_DEPTHS, 0.01, "const0")[7][0]
    mech = mechanisms(translate(c)[0])
    joint = _exact_joint(mech, [("d", 0), ("d", 1), ("o", 0)])
    exact_det0 = _exact_joint(mech, [("d", 0)])[1]
    hist = np.zeros(8)
    n = 0
    for chunk in range(10):
        np.random.seed(int(np.random.SeedSequence([9, chunk]).generate_state(1)[0]))
        dem_simulation._MODEL_CACHE.clear()
        _, payload = dem_simulation.simulate_circuit_with_dem(c, 400_001)
        d = np.array([x["data"] for x in payload["detectors"]]).T % 2
        o = np.asarray(payload["logicals"][0]["data"]) % 2
        hist += np.bincount(d[:, 0] + 2 * d[:, 1] + 4 * o, minlength=8)
        n += len(o)
    k0 = hist[1] + hist[3] + hist[5] + hist[7]
    z = (k0 / n - exact_det0) / math.sqrt(exact_det0 * (1 - exact_det0) / n)
    expected = joint * n
    chi2 = float(((hist - expected) ** 2 / expected).sum())
    lines = [f"det0: {k0 / n:.6f} vs exact {exact_det0:.6f}, n={n}, z={z:+.2f}, "
             f"p={2 * stats.norm.sf(abs(z)):.3f}",
             f"joint law (8 cells): chi2={chi2:.2f} dof=7 "
             f"p={stats.chi2.sf(chi2, 7):.3f}",
             f"observed: {hist.astype(int).tolist()}",
             f"expected: {np.round(expected).astype(int).tolist()}"]
    print("\n".join(lines))
    Path(args.out).mkdir(parents=True, exist_ok=True)
    (Path(args.out) / "followup_repo_dem.txt").write_text("\n".join(lines) + "\n")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("command", choices=("dem", "floor", "semantics",
                                            "hazards", "sample", "followup"))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--trials", type=int, default=300)
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--chunks", type=int, default=10)
    parser.add_argument("--chunk-shots", type=int, default=20_000)
    parser.add_argument("--stim-shots", type=int, default=2_000_000)
    parser.add_argument("--backend", choices=("sdim", "repo_dem"),
                        default="sdim",
                        help="sample: sdim's public Program.simulate, or the "
                             "repo's production DEM backend")
    args = parser.parse_args(argv)
    {"dem": cmd_dem, "floor": cmd_floor, "semantics": cmd_semantics,
     "hazards": cmd_hazards, "sample": cmd_sample,
     "followup": cmd_followup}[args.command](args)


if __name__ == "__main__":
    main()
