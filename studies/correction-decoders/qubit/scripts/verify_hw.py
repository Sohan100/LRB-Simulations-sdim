"""Adversarial verification of work package J (hardware protocol, DEM learning).

Independent of ``studies/correction-decoders/qubit/scripts/hardware_model.py`` wherever possible:

``graph``      Claims 1-3. My own networkx search: embed the 9-qubit check
               interaction graph in a square grid, count embeddings, check that
               each is a 3x3 block, and compute d(2, 3). IQM Garnet couplers are
               parsed from IQM's own qiskit-on-iqm ``fake_garnet.py`` (downloaded
               text, path given with ``--garnet-file``), not from J's constant.
``idle``       Claim 8. Pauli twirl of amplitude + phase damping from Kraus
               operators (my own Pauli-transfer-matrix computation), compared
               with the closed form and with J's ``pauli_twirl_idle``; the
               readout-window idle in units of the SI1000 p; sensitivity to the
               published T1/T2 inputs.
``checkfree``  Claim 9. Independent check-free (const0) circuits for the device
               models (my own layer compiler and noise insertion, written from
               J's documented model) and EXACT logical error rates from the
               detector error model (Walsh-Hadamard over the 3 outcome bits, no
               sampling). Decay fits with the same weighted log-linear fit.
               Also: the 4-CZ code-space gadget (see ``gadget``) and native CZ.
``gadget``     Claim 4. On the code space (S_Z2 = Z1 Z2 Z4 = +1), CZ(2,3) equals
               CZ(1,3) CZ(3,4); on the 3x3 patch both act through the single
               ancilla a8 (neighbours 1, 3, 4) in 4 CZ. Checks: equality on code
               states, inequality off the code space, ancilla restored,
               deterministic hardware LRB circuits with checks (J's builder with
               the gadget patched in at runtime), circuit-level distance 2.
``mld``        Claim 9 (MLD every step) at reduced statistics with J's builder and
               the lead's exact MLD, fresh seeds; also with the 4-CZ gadget and
               with noiseless, instantaneous check rounds (claim 10 limit).
``reset``      Claim 11: zero-latency reset with e_reset = 0 vs no reset; paired
               bootstrap of J's latency sweep.
``dem``        Claim 15: exact re-run of one ``dem_learning.study_circuit`` row
               (same seeds) against J's JSON; independent p_ij estimator; a
               decomposed-graph DEM as a fairer pairs-only baseline.

Outputs go to studies/correction-decoders/qubit/results/J_verify/.

    module load python/3.11
    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=$PWD/src
    python3 studies/correction-decoders/qubit/scripts/verify_hw.py graph --garnet-file <fake_garnet.py text>
    python3 studies/correction-decoders/qubit/scripts/verify_hw.py idle
    python3 studies/correction-decoders/qubit/scripts/verify_hw.py checkfree
    python3 studies/correction-decoders/qubit/scripts/verify_hw.py gadget
    python3 studies/correction-decoders/qubit/scripts/verify_hw.py mld --workers 16 --shots 20000
    python3 studies/correction-decoders/qubit/scripts/verify_hw.py reset --workers 16 --shots 20000
    python3 studies/correction-decoders/qubit/scripts/verify_hw.py dem --workers 4
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path

if __name__ == "__main__":
    _THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    if any(os.environ.get(_v) != "1" for _v in _THREAD_VARS):
        for _v in _THREAD_VARS:
            os.environ[_v] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import networkx as nx
import numpy as np
import stim
from networkx.algorithms import isomorphism

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "qubit512"))

from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

OUT = ROOT / "studies" / "correction-decoders" / "qubit" / "results" / "J_verify"
SEQ_SEED = 20261028
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]

# The check interaction graph exactly as given in the verification brief.
TANNER = [(0, 5), (1, 5), (3, 5), (1, 6), (2, 6), (4, 6),
          (0, 7), (1, 7), (2, 7), (1, 8), (3, 8), (4, 8)]


def _dump(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(obj, indent=1, default=float))
    return path


# ---------------------------------------------------------------------------
# Claims 1-3: graph search
# ---------------------------------------------------------------------------


def _monomorphisms(host: nx.Graph, pattern: nx.Graph) -> list[dict]:
    """All subgraph monomorphisms pattern -> host as {pattern: host}."""
    matcher = isomorphism.GraphMatcher(host, pattern)
    return [{p: h for h, p in m.items()}
            for m in matcher.subgraph_monomorphisms_iter()]


def parse_garnet_couplers(path: Path) -> list[tuple[str, str]]:
    """Read ``qubit_connectivity`` from IQM's fake_garnet.py as plain text."""
    text = path.read_text()
    block = text.split("qubit_connectivity = [", 1)[1].split("]\n    architecture", 1)[0]
    pairs = re.findall(r'\["(QB\d+)",\s*"(QB\d+)"\]', block)
    return [(a, b) for a, b in pairs]


def graph_checks(garnet_file: Path | None) -> dict:
    tanner = nx.Graph(TANNER)
    out: dict = {"tanner_edges": TANNER}
    # consistency with the lead's stabilizer definitions
    from_code = sorted((q, C.STABILIZER_ANCILLA[s])
                       for s in range(4)
                       for q, ch in enumerate(C.STABILIZER_PAULIS[s]) if ch != "_")
    out["matches_lrb_qubit_code"] = sorted(TANNER) == from_code
    out["degrees"] = dict(tanner.degree())
    out["isomorphic_to_3x3_grid"] = nx.is_isomorphic(tanner, nx.grid_2d_graph(3, 3))

    # Embeddings into a 7x7 grid: every placement and symmetry.
    n = 7
    host = nx.grid_2d_graph(n, n)
    embs = _monomorphisms(host, tanner)
    blocks, dists, graph_dists, boxes = set(), [], [], set()
    for e in embs:
        rows = [e[v][0] for v in range(9)]
        cols = [e[v][1] for v in range(9)]
        box = (max(rows) - min(rows) + 1, max(cols) - min(cols) + 1)
        boxes.add(box)
        blocks.add(frozenset(e.values()))
        (r2, c2), (r3, c3) = e[2], e[3]
        dists.append(abs(r2 - r3) + abs(c2 - c3))
        graph_dists.append(nx.shortest_path_length(host, e[2], e[3]))
    out["grid7_embeddings"] = len(embs)
    out["grid7_expected_25_positions_x_8"] = 25 * 8
    out["grid7_distinct_site_sets"] = len(blocks)
    out["grid7_bounding_boxes"] = sorted(boxes)
    out["grid7_manhattan_d23"] = sorted(set(dists))
    out["grid7_graph_d23"] = sorted(set(graph_dists))
    # the centre (degree 4) always maps to the block centre, 2 and 3 to
    # opposite corners
    corners = set()
    for e in embs:
        rmin = min(e[v][0] for v in range(9))
        cmin = min(e[v][1] for v in range(9))
        rel = {v: (e[v][0] - rmin, e[v][1] - cmin) for v in range(9)}
        corners.add((rel[1], tuple(sorted((rel[2], rel[3])))))
    out["centre_and_23_positions"] = sorted(map(str, corners))

    # Adding the fold coupling (2,3)
    with23 = nx.Graph(TANNER + [(2, 3)])
    out["with23_bipartite"] = nx.is_bipartite(with23)
    path = nx.shortest_path(tanner, 2, 3)
    out["shortest_23_path_in_tanner"] = path
    out["odd_cycle_through_23"] = path + [2]
    out["odd_cycle_length"] = len(path)
    out["grid7_embeddings_with23"] = len(_monomorphisms(host, with23))

    # Claim 3: the relabelling SWAP(2,3) is the patch reflection that also
    # swaps (5,7) and (6,8), i.e. Z-check and X-check ancillas.
    sigma = {0: 0, 1: 1, 4: 4, 2: 3, 3: 2, 5: 7, 7: 5, 6: 8, 8: 6}
    out["swap23_reflection_is_automorphism"] = all(
        tanner.has_edge(sigma[a], sigma[b]) for a, b in TANNER)
    # logical H = SWAP(2,3) then H^5: S_Z1 <-> S_X1, S_Z2 <-> S_X2; the ancilla
    # of the image stabilizer must be sigma(ancilla).
    swapped = {}
    for s in range(4):
        pauli = C.STABILIZER_PAULIS[s]
        support = {q for q, ch in enumerate(pauli) if ch != "_"}
        image_support = {sigma[q] for q in support}
        anc = C.STABILIZER_ANCILLA[s]
        nb = set(tanner.neighbors(sigma[anc]))
        swapped[C.STABILIZER_NAMES[s]] = {
            "image_support": sorted(image_support),
            "physical_ancilla_after_swap": sigma[anc],
            "ancilla_neighbours": sorted(nb),
            "ok": nb == image_support}
    out["frame1_roles"] = swapped

    if garnet_file is not None:
        couplers = parse_garnet_couplers(garnet_file)
        garnet = nx.Graph(couplers)
        out["garnet_source"] = str(garnet_file)
        out["garnet_num_couplers"] = len(couplers)
        out["garnet_num_qubits"] = garnet.number_of_nodes()
        out["garnet_bipartite"] = nx.is_bipartite(garnet)
        # Garnet is a subgraph of the square lattice: find one embedding
        lattice = nx.grid_2d_graph(7, 7)
        gm = isomorphism.GraphMatcher(lattice, garnet)
        coords = None
        for m in gm.subgraph_monomorphisms_iter():
            coords = {g: h for h, g in m.items()}
            break
        out["garnet_embeds_in_square_lattice"] = coords is not None
        embs = _monomorphisms(garnet, tanner)
        sets = {}
        for e in embs:
            key = frozenset(e.values())
            d = nx.shortest_path_length(garnet, e[2], e[3])
            sets.setdefault(key, []).append(d)
        out["garnet_tanner_embeddings"] = len(embs)
        out["garnet_patches"] = [
            {"qubits": sorted(k, key=lambda s: int(s[2:])),
             "labellings": len(v), "d23_all_labellings": sorted(set(v))}
            for k, v in sets.items()]
        out["garnet_embeddings_with23"] = len(_monomorphisms(garnet, with23))
        # agreement with J's transcription
        try:
            import hardware_model as H
            out["matches_J_GARNET_COUPLERS"] = (
                {frozenset(c) for c in couplers}
                == {frozenset(c) for c in H.GARNET_COUPLERS})
        except Exception as exc:  # pragma: no cover
            out["matches_J_GARNET_COUPLERS"] = f"error: {exc}"
        # medians of the sample error profile in the same file
        text = garnet_file.read_text()

        def _dict_block(opening):
            blk = text.split(opening, 1)[1].split("}", 1)[0]
            return [float(v) for v in re.findall(r":\s*([0-9.]+)", blk)]
        t1 = _dict_block("t1s={")
        t2 = _dict_block("t2s={")
        prx = _dict_block('"prx": {')
        cz_blk = text.split('"cz": {', 1)[1].split("}", 1)[0]
        cz = [float(v) for v in re.findall(r"\):\s*([0-9.]+)", cz_blk)]
        out["garnet_sample_profile_medians"] = {
            "t1_us": float(np.median(t1)) / 1000, "t2_us": float(np.median(t2)) / 1000,
            "prx_depol": float(np.median(prx)), "cz_depol": float(np.median(cz)),
            "n": [len(t1), len(t2), len(prx), len(cz)]}
    return out


# ---------------------------------------------------------------------------
# Claim 8: idle twirl and SI1000 mapping
# ---------------------------------------------------------------------------

PAULIS = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
          np.diag([1.0, -1.0])]


def twirl_from_kraus(t_us: float, t1_us: float, t2_us: float) -> np.ndarray:
    """Pauli probabilities (I, X, Y, Z) of the twirled T1/T2 channel.

    Built from Kraus operators (amplitude damping gamma = 1 - e^{-t/T1}
    followed by pure dephasing such that the coherence decays as e^{-t/T2});
    the twirl keeps the diagonal of the Pauli transfer matrix R_ii =
    tr(P_i E(P_i))/2, and p = H R_diag / 4 with the 4x4 sign matrix.
    """
    gamma = 1 - math.exp(-t_us / t1_us) if math.isfinite(t1_us) else 0.0
    coh = math.exp(-t_us / t2_us) if math.isfinite(t2_us) else 1.0
    lam = coh / math.sqrt(1 - gamma)          # extra dephasing factor (<= 1)
    q = (1 - lam) / 2
    kraus_ad = [np.array([[1, 0], [0, math.sqrt(1 - gamma)]]),
                np.array([[0, math.sqrt(gamma)], [0, 0]])]
    kraus_pd = [math.sqrt(1 - q) * np.eye(2), math.sqrt(q) * PAULIS[3]]
    kraus = [b @ a for a in kraus_ad for b in kraus_pd]

    def channel(rho):
        return sum(k @ rho @ k.conj().T for k in kraus)
    r = np.array([np.real(np.trace(P @ channel(P))) / 2 for P in PAULIS])
    signs = np.array([[1, 1, 1, 1], [1, 1, -1, -1], [1, -1, 1, -1],
                      [1, -1, -1, 1]])
    return signs @ r / 4


def twirl_closed_form(t_us, t1_us, t2_us):
    a = 1 - math.exp(-t_us / t1_us) if math.isfinite(t1_us) else 0.0
    b = 1 - math.exp(-t_us / t2_us) if math.isfinite(t2_us) else 0.0
    return a / 4, a / 4, b / 2 - a / 4


def idle_checks() -> dict:
    import hardware_model as H
    out = {"formula_vs_kraus": [], "vs_J": []}
    for t, t1, t2 in [(1.0, 40, 21.05), (0.5, 40, 21.05), (1.5, 40, 21.05),
                      (0.04, 30, 10), (2.0, 100, 100), (1.0, 27.42, 21.05),
                      (1.0, 40, 8.43)]:
        k = twirl_from_kraus(t, t1, t2)
        cf = twirl_closed_form(t, t1, t2)
        jj = H.pauli_twirl_idle(1000 * t, t1, t2)
        out["formula_vs_kraus"].append(
            {"t_us": t, "T1": t1, "T2": t2, "kraus_xyz": k[1:].tolist(),
             "closed_xyz": list(cf), "max_abs_diff": float(np.max(np.abs(
                 k[1:] - np.array(cf))))})
        out["vs_J"].append(float(np.max(np.abs(np.array(jj) - np.array(cf)))))
    p = 1.25 * 5e-3
    rows = []
    cases = [
        ("J main: T1=40 (IQM 'order of 40 us'), T2e=21.05 (Sharma Tab.3), 1.0 us", 1.0, 40, 21.05),
        ("readout pulse only (800 ns, no assumed 200 ns)", 0.8, 40, 21.05),
        ("Sharma snapshot consistent: T1=27.42, T2e=21.05, 1.0 us", 1.0, 27.42, 21.05),
        ("no DD: T1=40, T2 Ramsey=8.43, 1.0 us", 1.0, 40, 8.43),
        ("fake-Garnet sample profile medians (T1 37.7, T2 9.6), 1.0 us", 1.0, 37.7, 9.6),
        ("reset window 0.5 us (J main)", 0.5, 40, 21.05),
        ("readout + reset window 1.5 us (J main)", 1.5, 40, 21.05),
        ("T1=T2=100 us, 1.0 us", 1.0, 100, 100),
    ]
    for label, t, t1, t2 in cases:
        tot = sum(twirl_closed_form(t, t1, t2))
        rows.append({"case": label, "total_pauli": tot, "over_p": tot / p,
                     "linearised": t * (1 / (4 * t1) + 1 / (2 * t2))})
    out["si1000_p"] = p
    out["mapping_cases"] = rows
    return out


# ---------------------------------------------------------------------------
# Claim 9: independent check-free circuits with exact error rates
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Dev:
    name: str
    t1_us: float
    t2_us: float
    t1q_ns: float
    t2q_ns: float
    r1q: float
    r2q: float
    extra_2q: float = 0.0     # extra depolarizing per 2q layer on idle qubits
    cz23: str = "routed"      # routed | native | alt4


DEVS = {
    "iqm_garnet": Dev("iqm_garnet", 40.0, 21.05, 20.0, 40.0, 9e-4, 5e-3),
    "iqm_t100": Dev("iqm_t100", 100.0, 100.0, 20.0, 40.0, 9e-4, 5e-3),
    "iqm_garnet_native23": Dev("iqm_garnet_native23", 40.0, 21.05, 20.0, 40.0,
                               9e-4, 5e-3, cz23="native"),
    "ion_helios": Dev("ion_helios", math.inf, math.inf, 1e4, 7e4, 2.5e-5,
                      7.9e-4, extra_2q=5e-4, cz23="native"),
    "atom_rb": Dev("atom_rb", math.inf, 1.5e6, 1e3, 2e5, 8.8e-4, 5e-3,
                   cz23="native"),
}
PERM = {0: (0, 1, 2, 3, 4), 1: (0, 1, 3, 2, 4)}


def _layers_routed8():
    """J's documented routed CZ(2,3): copy z2->a7, z3->a5, CZ(a7,a5) bridged
    by data 0, uncopy (13 layers after merging adjacent 1q layers)."""
    return [("1q", [("H", 7), ("H", 5)]),
            ("2q", [(2, 7), (3, 5)]),
            ("1q", [("H", 7), ("H", 5), ("H", 0)]),
            ("2q", [(7, 0)]), ("1q", [("H", 0)]),
            ("2q", [(0, 5)]), ("1q", [("H", 0)]),
            ("2q", [(7, 0)]), ("1q", [("H", 0)]),
            ("2q", [(0, 5)]),
            ("1q", [("H", 7), ("H", 5)]),
            ("2q", [(2, 7), (3, 5)]),
            ("1q", [("H", 7), ("H", 5)]),
            ("release", [7, 5])]


def alt4_ancilla(frame: int) -> int:
    """Ancilla adjacent to physical {1, perm[3], 4}: a8 (frame 0) or a6."""
    return 8 if frame == 0 else 6


def _layers_alt4(frame: int):
    """CZ(1,3) CZ(3,4) (= CZ(2,3) on the S_Z2 = +1 code space) via one ancilla:
    CNOT(3 -> a), CZ(a, 1), CZ(a, 4), CNOT(3 -> a). 4 CZ, 8 layers."""
    a, p3 = alt4_ancilla(frame), PERM[frame][3]
    return [("1q", [("H", a)]), ("2q", [(p3, a)]), ("1q", [("H", a)]),
            ("2q", [(a, 1)]), ("2q", [(a, 4)]),
            ("1q", [("H", a)]), ("2q", [(p3, a)]), ("1q", [("H", a)]),
            ("release", [a])]


def _layers_par5(frame: int):
    """CZ(1,3) CZ(3,4) via the parity copy: a <- z1 + z4 (= z2 on the code
    space), CZ(a, 3), uncopy. 5 CZ, 9 layers. A Z error on a returns to the
    data as Z1 Z4 ~ Z2 (weight-1 class), so no single fault is a logical."""
    a, p3 = alt4_ancilla(frame), PERM[frame][3]
    return [("1q", [("H", a)]), ("2q", [(1, a)]), ("2q", [(4, a)]),
            ("1q", [("H", a)]), ("2q", [(a, p3)]),
            ("1q", [("H", a)]), ("2q", [(1, a)]), ("2q", [(4, a)]),
            ("1q", [("H", a)]), ("release", [a])]


def _merge_1q(layers):
    """Merge adjacent 1q layers; cancel H.H on shared qubits (J's rule)."""
    out = []
    for kind, ops in layers:
        if out and kind == "1q" and out[-1][0] == "1q":
            prev = out[-1][1]
            prev_q = {q: g for g, q in prev}
            shared = [q for g, q in ops if q in prev_q]
            if all(prev_q[q] == "H" and dict((qq, gg) for gg, qq in ops)[q] == "H"
                   for q in shared):
                cancel = set(shared)
                merged = ([o for o in prev if o[1] not in cancel]
                          + [o for o in ops if o[1] not in cancel])
                if merged:
                    out[-1] = ("1q", merged)
                else:
                    out.pop()
                continue
        out.append((kind, list(ops)))
    return out


def compile_clifford(descriptor: str, inverse: bool, frame: int, cz23: str):
    """Physical layers of one logical Clifford, following J's documented
    compilation: H_L = one layer of 5 pi/2 pulses + frame toggle (virtual
    SWAP); S, S_dag, Z virtual; X, Z Pauli frame (virtual); CZ(2,3) routed,
    native or the 4-CZ code-space gadget. Virtual ops are ("virtual", ...)
    layers (zero time, noiseless) and, as in J's compiler, they separate 1q
    layers for merging purposes."""
    tokens = (C.clifford_inverse_sequence(descriptor) if inverse
              else C.clifford_gate_sequence(descriptor))
    layers = []
    pending, kind = [], None

    def flush():
        nonlocal pending, kind
        if pending:
            layers.append((kind, pending))
        pending, kind = [], None

    for token in tokens:
        for gate, targets in C.logical_gate_operations(token):
            perm = PERM[frame]
            if gate == "SWAP":
                frame ^= 1
                continue
            if gate == "I":
                continue
            if gate == "CZ":
                flush()
                if cz23 == "routed":
                    layers += [(k, list(o)) for k, o in _layers_routed8()]
                elif cz23 == "native":
                    layers.append(("2q", [(2, 3)]))
                elif cz23 == "alt4":
                    layers += [(k, list(o)) for k, o in _layers_alt4(frame)]
                elif cz23 == "par5":
                    layers += [(k, list(o)) for k, o in _layers_par5(frame)]
                else:
                    raise ValueError(cz23)
                continue
            q = perm[targets[0]]
            op_kind = "1q" if gate == "H" else "virtual"
            if kind != op_kind or any(q == t for _, t in pending):
                flush()
                kind = op_kind
            pending.append((gate, q))
    flush()
    return _merge_1q(layers), frame


def idle_xyz(dev: Dev, t_ns: float):
    return twirl_closed_form(t_ns / 1000.0, dev.t1_us, dev.t2_us)


def build_checkfree(sequence, depth: int, dev: Dev, noisy: bool = True,
                    fresh_ancilla: bool = False):
    """Encoding (ideal), m Cliffords + mirror inverses, ideal X readout of all
    five data; detectors = final X stabilizers; observable = x0 + x3."""
    c = stim.Circuit()
    c.append("I", [8])
    for gate, targets in C.encoding_operations():
        c.append(gate, list(targets))
    active = set(range(5))
    p1 = 1.5 * dev.r1q
    p2 = 1.25 * dev.r2q
    frame = 0
    words = [(d, False) for d in sequence[:depth]] + [
        (d, True) for d in reversed(sequence[:depth])]
    n_cz = 0
    t_total = 0.0
    for descriptor, inverse in words:
        layers, frame = compile_clifford(descriptor, inverse, frame, dev.cz23)
        for kind, ops in layers:
            if kind == "virtual":
                for g, q in ops:
                    c.append(g, [q])
                continue
            if kind == "release":
                active -= set(ops)
                if fresh_ancilla:   # noiseless, instantaneous reset (diagnostic)
                    c.append("R", sorted(ops))
                continue
            if kind == "1q":
                acted = {q for _, q in ops}
                for g, q in ops:
                    c.append(g, [q])
                    if noisy and p1 > 0:
                        c.append("DEPOLARIZE1", [q], p1)
                dur = dev.t1q_ns
            else:
                acted = {q for pair in ops for q in pair}
                for a, b in ops:
                    c.append("CZ", [a, b])
                    n_cz += 1
                    if noisy and p2 > 0:
                        c.append("DEPOLARIZE2", [a, b], p2)
                dur = dev.t2q_ns
            active |= acted
            t_total += dur
            if noisy:
                px, py, pz = idle_xyz(dev, dur)
                for q in sorted(active - acted):
                    if px + py + pz > 0:
                        c.append("PAULI_CHANNEL_1", [q], [px, py, pz])
                    if kind == "2q" and dev.extra_2q > 0:
                        c.append("DEPOLARIZE1", [q], dev.extra_2q)
    assert frame == 0, "mirror circuit must end in frame 0"
    for q in range(5):
        c.append("H", [q])
    c.append("M", list(range(5)))
    rec = {q: q - 5 for q in range(5)}  # target_rec offsets of the 5 results
    c.append("DETECTOR", [stim.target_rec(rec[q]) for q in (0, 1, 2)])
    c.append("DETECTOR", [stim.target_rec(rec[q]) for q in (1, 3, 4)])
    c.append("OBSERVABLE_INCLUDE", [stim.target_rec(rec[q]) for q in (0, 3)], 0)
    return c, n_cz, t_total


def exact_joint(dem: stim.DetectorErrorModel) -> np.ndarray:
    """P(x) over the (detectors, observable 0) bits, exact for independent
    mechanisms: characters chi(u) = prod (1 - 2 p_e)^{[u . f_e odd]}."""
    n_det = dem.num_detectors
    nbits = n_det + 1
    size = 1 << nbits
    us = np.arange(size)
    chars = np.ones(size)
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        mask = 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                mask ^= 1 << t.val
            elif t.is_logical_observable_id():
                mask ^= 1 << (n_det + t.val)
        if mask == 0 or p == 0:
            continue
        odd = np.array([bin(u & mask).count("1") & 1 for u in us], bool)
        chars[odd] *= (1 - 2 * p)
    signs = np.array([[(-1) ** (bin(u & x).count("1") & 1) for u in us]
                      for x in us], float)
    return signs @ chars / size


def checkfree_point(sequence, depth, dev):
    c, n_cz, t_ns = build_checkfree(sequence, depth, dev)
    dem = c.detector_error_model()
    P = exact_joint(dem)
    n_det = dem.num_detectors
    obs_bit = 1 << n_det
    err = sum(min(P[s], P[s | obs_bit]) for s in range(1 << n_det))
    acc = P[0] + P[obs_bit]
    return {"F_mld": 1 - 2 * err, "F_det": abs(P[0] - P[obs_bit]) / acc,
            "acceptance": acc, "n_cz": n_cz, "t_ns": t_ns}


def weighted_fit(depths, F, n_boot=400, seed=7):
    """F(m) = b f^m on the mean over sequences: weighted least squares on
    log F, weights (mean/sem)^2 with sem floored at 1e-4 (as in J/lead);
    bootstrap over sequences. Also an unweighted fit for comparison."""
    depths = np.asarray(depths, float)

    def fit(sample, weighted=True):
        mean = np.nanmean(sample, 0)
        sem = np.maximum(np.nanstd(sample, 0) / np.sqrt(sample.shape[0]), 1e-4)
        ok = (mean > 0) & (mean < 1)
        x, y = depths[ok], np.log(mean[ok])
        w = (mean[ok] / sem[ok]) ** 2 if weighted else np.ones(ok.sum())
        xm, ym = (w * x).sum() / w.sum(), (w * y).sum() / w.sum()
        slope = (w * (x - xm) * (y - ym)).sum() / (w * (x - xm) ** 2).sum()
        return math.exp(ym - slope * xm), math.exp(slope)
    b, f = fit(F)
    rng = np.random.default_rng(seed)
    boots = np.array([fit(F[rng.integers(0, F.shape[0], F.shape[0])])
                      for _ in range(n_boot)])
    bu, fu = fit(F, weighted=False)
    return {"b": b, "f": f, "f_err": float(np.std(boots[:, 1])),
            "b_err": float(np.std(boots[:, 0])), "f_unweighted": fu,
            "F_mean": np.nanmean(F, 0).tolist()}


def checkfree_study(num_seq: int = 30) -> dict:
    seqs = E.clifford_sequences(num_seq, max(DEPTHS), SEQ_SEED)
    out = {"num_seq": num_seq, "depths": DEPTHS, "models": {}}
    # noiseless determinism of the independent builder (all variants)
    det_ok = True
    for cz in ("routed", "native", "alt4", "par5"):
        dev = replace(DEVS["iqm_garnet"], cz23=cz)
        for seq in seqs[:6]:
            for depth in (0, 1, 3, 9, 22):
                c, _, _ = build_checkfree(seq, depth, dev, noisy=False)
                d, o = c.compile_detector_sampler(seed=1).sample(
                    64, separate_observables=True)
                det_ok &= (not d.any()) and (not o.any())
    out["noiseless_deterministic_all_variants"] = bool(det_ok)
    # exact vs sampled sanity check of the enumeration (one circuit)
    c, _, _ = build_checkfree(seqs[0], 6, DEVS["iqm_garnet"])
    P = exact_joint(c.detector_error_model())
    d, o = c.compile_detector_sampler(seed=5).sample(
        400_000, separate_observables=True)
    x = d[:, 0].astype(int) + 2 * d[:, 1].astype(int) + 4 * o[:, 0].astype(int)
    emp = np.bincount(x, minlength=8) / len(x)
    out["exact_vs_sampled"] = {"exact": P.tolist(), "sampled_4e5": emp.tolist(),
                               "max_abs_diff": float(np.max(np.abs(P - emp)))}
    variants = [("iqm_garnet", "routed"), ("iqm_garnet", "alt4"),
                ("iqm_garnet", "par5"), ("iqm_garnet", "native"),
                ("iqm_t100", "routed"), ("iqm_t100", "alt4"),
                ("iqm_t100", "par5"), ("ion_helios", "native"),
                ("atom_rb", "native")]
    for name, cz in variants:
        dev = replace(DEVS[name], cz23=cz)
        Fm = np.zeros((num_seq, len(DEPTHS)))
        Fd = np.zeros_like(Fm)
        acc = np.zeros_like(Fm)
        ncz = np.zeros_like(Fm)
        for si, seq in enumerate(seqs):
            for di, depth in enumerate(DEPTHS):
                r = checkfree_point(seq, depth, dev)
                Fm[si, di], Fd[si, di] = r["F_mld"], r["F_det"]
                acc[si, di], ncz[si, di] = r["acceptance"], r["n_cz"]
        key = f"{name}/{cz}"
        out["models"][key] = {
            "const0_mld": weighted_fit(DEPTHS, Fm),
            "const0": weighted_fit(DEPTHS, Fd),
            "rejection_mean": (1 - acc.mean(0)).tolist(),
            "mean_cz_per_depth_step": float(
                np.polyfit(DEPTHS, ncz.mean(0), 1)[0])}
    return out


# ---------------------------------------------------------------------------
# Claim 4: the 4-CZ code-space gadget
# ---------------------------------------------------------------------------


def _alt4_layers_J(frame: int):
    import hardware_model as H
    a, p3 = alt4_ancilla(frame), PERM[frame][3]
    L = H.Layer
    return H.compact([
        L("1q", (("H", (a,)),)), L("2q", (("CZ", (p3, a)),)),
        L("1q", (("H", (a,)),)), L("2q", (("CZ", (a, 1)),)),
        L("2q", (("CZ", (a, 4)),)), L("1q", (("H", (a,)),)),
        L("2q", (("CZ", (p3, a)),)), L("1q", (("H", (a,)),)),
        # unreset ancilla holding m: byproduct (Z1 Z4)^m -> Pauli frame
        L("feedback", (("Z", (a, 1)), ("Z", (a, 4)))),
        L("release", (("I", (a,)),))])


def _par5_layers_J(frame: int):
    import hardware_model as H
    a, p3 = alt4_ancilla(frame), PERM[frame][3]
    L = H.Layer
    return H.compact([
        L("1q", (("H", (a,)),)), L("2q", (("CZ", (1, a)),)),
        L("2q", (("CZ", (4, a)),)), L("1q", (("H", (a,)),)),
        L("2q", (("CZ", (a, p3)),)), L("1q", (("H", (a,)),)),
        L("2q", (("CZ", (1, a)),)), L("2q", (("CZ", (4, a)),)),
        L("1q", (("H", (a,)),)),
        # unreset ancilla holding m: byproduct Z(phys p3)^m -> Pauli frame
        L("feedback", (("Z", (a, p3)),)),
        L("release", (("I", (a,)),))])


@contextlib.contextmanager
def patched_alt4(gadget: str = "alt4"):
    """Run J's builder with CZ(2,3) replaced by CZ(1,3) CZ(3,4) everywhere:
    the code-level ops (used for detector back-propagation) AND the physical
    compilation (4-CZ gadget through the frame's S_X2 ancilla). Nothing is
    written to disk; the patch is undone on exit."""
    import hardware_model as H
    orig_ops = C.logical_clifford_operations
    orig_compile = H.compile_logical_ops

    def ops_alt(descriptor, inverse=False):
        out = []
        for g, t in orig_ops(descriptor, inverse):
            if g == "CZ" and tuple(t) == (2, 3):
                out += [("CZ", (1, 3)), ("CZ", (3, 4))]
            else:
                out.append((g, t))
        return out

    def compile_alt(code_ops, frame, *, cz23="routed", bridge=0,
                    virtual_paulis=True, virtual_swap=True):
        layers, pending, kind = [], [], None

        def flush():
            nonlocal pending, kind
            if pending:
                layers.append(H.Layer(kind, tuple(pending)))
            pending, kind = [], None
        ops = list(code_ops)
        i = 0
        while i < len(ops):
            gate, targets = ops[i]
            perm = H.PERM[frame]
            if gate == "SWAP":
                frame ^= 1
            elif gate == "CZ":
                assert tuple(targets) == (1, 3) and ops[i + 1] == ("CZ", (3, 4))
                flush()
                layers += (_alt4_layers_J(frame) if gadget == "alt4"
                           else _par5_layers_J(frame))
                i += 1
            elif gate != "I":
                q = perm[targets[0]]
                op_kind = ("virtual" if gate in H.VIRTUAL_GATES
                           or (gate == "X" and virtual_paulis) else "1q")
                if kind != op_kind or any(q in t for _, t in pending):
                    flush()
                    kind = op_kind
                pending.append((gate, (q,)))
            i += 1
        flush()
        return H.compact(layers), frame

    C.logical_clifford_operations = ops_alt
    H.compile_logical_ops = compile_alt
    try:
        yield
    finally:
        C.logical_clifford_operations = orig_ops
        H.compile_logical_ops = orig_compile


def gadget_checks() -> dict:
    import hardware_model as H
    out = {}
    enc = stim.Circuit()
    enc.append("I", [8])
    for g, t in C.encoding_operations():
        enc.append(g, list(t))
    seqs = E.clifford_sequences(8, 22, SEQ_SEED)
    rng = np.random.default_rng(3)

    def gadget_circuit(frame, fn=_layers_alt4):
        c = stim.Circuit()
        c.append("I", [8])
        for kind, ops in fn(frame):
            if kind == "1q":
                for g, q in ops:
                    c.append(g, [q])
            elif kind == "2q":
                for a, b in ops:
                    c.append("CZ", [a, b])
        return c

    # (a) on code states: gadget == CZ(2,3) (frame 0) and the ancilla returns
    # to |0>; logical states from random logical Clifford words.
    ok_code = True
    for trial in range(60):
        words = [C.SINGLE_QUBIT_CLIFFORDS[int(i)] for i in rng.integers(0, 24, 3)]
        prep = enc.copy()
        for w in words:
            for g, t in C.logical_clifford_operations(w):
                prep.append(g, list(t))
        sim = stim.TableauSimulator()
        sim.do(prep)
        sim.do(gadget_circuit(0))
        sim.do(stim.Circuit("CZ 2 3"))      # undo the intended gate
        sim.do(prep.inverse())
        ok_code &= all(sim.peek_z(q) == 1 for q in range(9))
    out["alt4_equals_CZ23_on_code_states_frame0"] = bool(ok_code)
    # frame 1: data 2 and 3 have swapped sites; the gadget uses a6 and phys 2
    ok_f1 = True
    for trial in range(60):
        words = [C.SINGLE_QUBIT_CLIFFORDS[int(i)] for i in rng.integers(0, 24, 3)]
        prep = enc.copy()
        for w in words:
            for g, t in C.logical_clifford_operations(w):
                prep.append(g, list(t))
        prep.append("SWAP", [2, 3])     # physical relabelling into frame 1
        sim = stim.TableauSimulator()
        sim.do(prep)
        sim.do(gadget_circuit(1))
        sim.do(stim.Circuit("CZ 2 3"))
        sim.do(prep.inverse())
        ok_f1 &= all(sim.peek_z(q) == 1 for q in range(9))
    out["alt4_equals_CZ23_on_code_states_frame1"] = bool(ok_f1)
    # (b) off the code space it is NOT CZ(2,3) (it is CZ(2,3) times a
    # Z3-controlled S_Z2): random 5-qubit stabilizer states
    differs = 0
    for trial in range(40):
        tab = stim.Tableau.random(5)
        sim = stim.TableauSimulator()
        sim.do_tableau(tab, list(range(5)))
        sim.do(gadget_circuit(0))
        sim.do(stim.Circuit("CZ 2 3"))
        sim.do_tableau(tab.inverse(), list(range(5)))
        differs += not all(sim.peek_z(q) == 1 for q in range(9))
    out["alt4_differs_off_code_space_of_40"] = differs
    # exact data unitary identity: gadget (ancilla in |0>) == CZ(1,3) CZ(3,4)
    same = True
    for trial in range(30):
        tab = stim.Tableau.random(5)
        s = stim.TableauSimulator()
        s.do_tableau(tab, list(range(5)))
        s.do(gadget_circuit(0))
        s.do(stim.Circuit("CZ 1 3\nCZ 3 4"))
        s.do_tableau(tab.inverse(), list(range(5)))
        same &= all(s.peek_z(q) == 1 for q in range(9))
    out["alt4_data_unitary_is_CZ13_CZ34_exactly"] = bool(same)
    # the same three checks for the 5-CZ parity gadget
    par = {"code_frame0": True, "code_frame1": True, "data_unitary": True}
    for frame in (0, 1):
        for trial in range(60):
            words = [C.SINGLE_QUBIT_CLIFFORDS[int(i)]
                     for i in rng.integers(0, 24, 3)]
            prep = enc.copy()
            for w in words:
                for g, t in C.logical_clifford_operations(w):
                    prep.append(g, list(t))
            if frame == 1:
                prep.append("SWAP", [2, 3])
            sim = stim.TableauSimulator()
            sim.do(prep)
            sim.do(gadget_circuit(frame, _layers_par5))
            sim.do(stim.Circuit("CZ 2 3"))
            sim.do(prep.inverse())
            par[f"code_frame{frame}"] &= all(sim.peek_z(q) == 1 for q in range(9))
    for trial in range(30):
        tab = stim.Tableau.random(5)
        sim = stim.TableauSimulator()
        sim.do_tableau(tab, list(range(5)))
        sim.do(gadget_circuit(0, _layers_par5))
        sim.do(stim.Circuit("CZ 1 3\nCZ 3 4"))
        sim.do_tableau(tab.inverse(), list(range(5)))
        par["data_unitary"] &= all(sim.peek_z(q) == 1 for q in range(9))
    out["par5_checks"] = {k: bool(v) for k, v in par.items()}
    out["par5_uses_only_patch_couplers"] = all(
        frozenset(pr) in {frozenset(e) for e in TANNER}
        for f in (0, 1) for k, ops in _layers_par5(f) if k == "2q" for pr in ops)
    out["alt4_counts"] = {"cz": 4, "pulses_1q": 4, "layers": "4 x 2q + 4 x 1q",
                          "duration_ns_iqm": 4 * 40 + 4 * 20,
                          "routed8_duration_ns_iqm": 6 * 40 + 7 * 20}
    # patch couplers used
    allowed = {frozenset(e) for e in TANNER}
    out["alt4_uses_only_patch_couplers"] = all(
        frozenset(p) in allowed for f in (0, 1) for k, ops in _layers_alt4(f)
        if k == "2q" for p in ops)

    # (c) J's builder with the gadget patched in: noiseless determinism with
    # checks (both terminals, reset and no-reset), and circuit distance 2.
    det = {}
    with patched_alt4("par5"):
        ok = True
        for kwargs in ({}, {"reset": False}):
            for seq in seqs[:4]:
                for depth in (0, 1, 2, 5, 9, 22):
                    for terminal in ("data5", "data2"):
                        lc = H.build_hw_lrb_circuit(seq, depth, H.IQM_GARNET,
                                                    terminal=terminal,
                                                    noisy=False, **kwargs)
                        d, o = lc.circuit.compile_detector_sampler(
                            seed=3).sample(32, separate_observables=True)
                        ok &= (not d.any()) and (not o.any())
        out["par5_J_builder_noiseless_deterministic"] = bool(ok)
    with patched_alt4():
        for kwargs in ({}, {"reset": False}):
            ok = True
            for seq in seqs[:4]:
                for depth in (0, 1, 2, 5, 9, 22):
                    for terminal in ("data5", "data2"):
                        lc = H.build_hw_lrb_circuit(seq, depth, H.IQM_GARNET,
                                                    terminal=terminal,
                                                    noisy=False, **kwargs)
                        d, o = lc.circuit.compile_detector_sampler(
                            seed=3).sample(32, separate_observables=True)
                        ok &= (not d.any()) and (not o.any())
            det[f"reset={kwargs.get('reset', True)}"] = bool(ok)
        out["alt4_J_builder_noiseless_deterministic"] = det
        dist = {}
        for depth in (1, 2, 4):
            lc = H.build_hw_lrb_circuit(seqs[1], depth, H.IQM_GARNET)
            err = lc.circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=4,
                dont_explore_edges_with_degree_above=9999,
                dont_explore_edges_increasing_symptom_degree=False)
            dist[f"alt4_m{depth}"] = len(err)
            lc0 = H.build_hw_const0_circuit(seqs[1], depth, H.IQM_GARNET)
            err0 = lc0.circuit.search_for_undetectable_logical_errors(
                dont_explore_detection_event_sets_with_size_above=4,
                dont_explore_edges_with_degree_above=9999,
                dont_explore_edges_increasing_symptom_degree=False)
            dist[f"alt4_const0_m{depth}"] = len(err0)
        # does the sequence actually contain logical S gates?
        dist["seq1_first4"] = seqs[1][:4]
    for depth in (1, 2, 4):
        lc = H.build_hw_lrb_circuit(seqs[1], depth, H.IQM_GARNET)
        err = lc.circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=4,
            dont_explore_edges_with_degree_above=9999,
            dont_explore_edges_increasing_symptom_degree=False)
        dist[f"routed8_m{depth}"] = len(err)
    out["circuit_distance"] = dist
    return out


def _undetectable_logical_weight(dem: stim.DetectorErrorModel) -> float:
    """First-order probability of faults with no detection event that flip
    the logical observable (XOR-combined over such mechanisms)."""
    q = 1.0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        tg = inst.targets_copy()
        if any(t.is_relative_detector_id() for t in tg):
            continue
        if any(t.is_logical_observable_id() for t in tg):
            q *= 1 - 2 * inst.args_copy()[0]
    return (1 - q) / 2


def _distance(circuit: stim.Circuit) -> int:
    """Length of the shortest undetectable logical error found by stim
    (99 if none exists, e.g. a noiseless Pauli-frame-only circuit)."""
    try:
        return len(circuit.search_for_undetectable_logical_errors(
            dont_explore_detection_event_sets_with_size_above=4,
            dont_explore_edges_with_degree_above=9999,
            dont_explore_edges_increasing_symptom_degree=False))
    except ValueError:
        return 99


def distance_study(num_seq: int = 10) -> dict:
    """Circuit-level distance and first-order undetectable logical weight of
    J's hardware circuits (with checks, data5 readout; and check-free) for the
    routed 8-CZ gadget, the 4-CZ and 5-CZ code-space gadgets, native CZ(2,3),
    and the lead's SI1000 circuit (reference)."""
    import hardware_model as H
    from lrb_qubit import circuits as Q
    seqs = E.clifford_sequences(num_seq, 22, SEQ_SEED)
    out = {"num_seq": num_seq, "variants": {}}
    depths_d = (1, 2, 4, 6)
    depths_w = (0, 2, 4, 6, 10, 14, 22)
    variants = {
        "routed8": (H.IQM_GARNET, contextlib.nullcontext),
        "alt4": (H.IQM_GARNET, patched_alt4),
        "par5": (H.IQM_GARNET, lambda: patched_alt4("par5")),
        "native": (H.DEVICES["iqm_garnet_native23"], contextlib.nullcontext),
    }
    for name, (dev, ctx) in variants.items():
        res = {"distance_checks": {}, "distance_const0": {},
               "U1_checks": [], "U1_const0": []}
        with ctx():
            for depth in depths_d:
                dc, d0 = [], []
                for seq in seqs[:4]:
                    dc.append(_distance(H.build_hw_lrb_circuit(seq, depth, dev).circuit))
                    d0.append(_distance(H.build_hw_const0_circuit(seq, depth, dev).circuit))
                res["distance_checks"][depth] = min(dc)
                res["distance_const0"][depth] = min(d0)
            for depth in depths_w:
                wc, w0 = [], []
                for seq in seqs:
                    wc.append(_undetectable_logical_weight(
                        H.build_hw_lrb_circuit(seq, depth, dev).circuit.detector_error_model()))
                    w0.append(_undetectable_logical_weight(
                        H.build_hw_const0_circuit(seq, depth, dev).circuit.detector_error_model()))
                res["U1_checks"].append(float(np.mean(wc)))
                res["U1_const0"].append(float(np.mean(w0)))
        res["U1_checks_slope_per_step"] = float(np.polyfit(depths_w, res["U1_checks"], 1)[0])
        res["U1_const0_slope_per_step"] = float(np.polyfit(depths_w, res["U1_const0"], 1)[0])
        out["variants"][name] = res
    # cross-check with my own (independent) check-free builder
    mine = {}
    for cz in ("routed", "alt4", "par5", "native"):
        dev = replace(DEVS["iqm_garnet"], cz23=cz)
        u1 = [float(np.mean([_undetectable_logical_weight(
            build_checkfree(seq, depth, dev)[0].detector_error_model())
            for seq in seqs])) for depth in depths_w]
        dmin = min(_distance(build_checkfree(seq, depth, dev)[0])
                   for seq in seqs[:4] for depth in (1, 2, 4))
        u1f = [float(np.mean([_undetectable_logical_weight(
            build_checkfree(seq, depth, dev, fresh_ancilla=True)[0]
            .detector_error_model()) for seq in seqs])) for depth in depths_w]
        dminf = min(_distance(build_checkfree(seq, depth, dev,
                                              fresh_ancilla=True)[0])
                    for seq in seqs[:4] for depth in (1, 2, 4))
        mine[cz] = {"U1_const0": u1, "slope_per_step": float(
            np.polyfit(depths_w, u1, 1)[0]), "distance_min_m124": dmin,
            "fresh_ancilla_U1_const0": u1f,
            "fresh_ancilla_slope_per_step": float(np.polyfit(depths_w, u1f, 1)[0]),
            "fresh_ancilla_distance_min_m124": dminf}
    out["my_builder_const0"] = mine
    # the lead's SI1000 circuit at the IQM-mapped p, for reference
    lc = Q.build_lrb_circuit(seqs[1], 4, 6.25e-3, terminal="data5")
    out["lead_si1000_distance_m4"] = _distance(lc.circuit)
    out["depths_w"] = list(depths_w)
    return out


# ---------------------------------------------------------------------------
# Claim 9 (MLD every step), claim 10 limit, with J's builder
# ---------------------------------------------------------------------------


class _CheckLayers(list):
    """Marker for check-round layers (made noiseless and instantaneous)."""


@contextlib.contextmanager
def patched_clean_checks():
    """Check rounds without any noise (gates, idles, SPAM): the data see
    exactly the noise of the check-free circuit, plus syndrome information."""
    import hardware_model as H
    orig_crl = H.check_round_layers
    orig_emit_all = H._Emitter.emit_all

    def crl(frame, reset=True):
        return _CheckLayers(orig_crl(frame, reset))

    def emit_all(self, layers):
        if isinstance(layers, _CheckLayers):
            saved = self.noisy
            self.noisy = False
            try:
                return orig_emit_all(self, layers)
            finally:
                self.noisy = saved
        return orig_emit_all(self, layers)
    H.check_round_layers = crl
    H._Emitter.emit_all = emit_all
    try:
        yield
    finally:
        H.check_round_layers = orig_crl
        H._Emitter.emit_all = orig_emit_all


def _mld_task(args):
    variant, si, seq, depths, shots, seed, device_name, extra = args
    import hardware_model as H
    dev = H.DEVICES[device_name]
    if extra.get("e_reset") is not None:
        dev = replace(dev, e_reset=extra["e_reset"])
    if extra.get("t_reset_ns") is not None:
        dev = replace(dev, t_reset_ns=extra["t_reset_ns"])
    kw = {"builder_kwargs": dict(extra.get("builder_kwargs", {}))}
    ctx = {"routed": contextlib.nullcontext, "alt4": patched_alt4,
           "par5": lambda: patched_alt4("par5"),
           "clean_checks": patched_clean_checks}[variant]
    t0 = time.perf_counter()
    with ctx():
        res = H.run_hw_point(dev, [seq], depths, shots, decode_shots=shots,
                             const_checks=(0, 1), unif_checks=(1,),
                             with_mld2=False, seed=seed + 1_000_000 * si,
                             check_every=extra.get("check_every"), **kw)
    return f"{variant}/{device_name}", si, res, time.perf_counter() - t0


def _run_pool(tasks, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    parts = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_mld_task, t) for t in tasks]
        for fut in as_completed(futs):
            variant, si, res, sec = fut.result()
            parts.setdefault(variant, {})[si] = res
    return parts


def _fits_from_parts(parts_v, depths, methods):
    import hardware_model as H
    n = len(parts_v)
    merged = H._merge(parts_v, 0.0, depths, n)
    out = {}
    for m in methods:
        if m in merged.tables:
            fit = H.decay_fit(merged.tables[m], depths)
            fid = merged.tables[m].fidelity()
            out[m] = {"f": fit["f"], "f_err": fit["f_err"], "b": fit["b"],
                      "F_mean": np.nanmean(fid, 0).tolist(),
                      "my_fit": weighted_fit(depths, fid)["f"]}
    return out, merged


def mld_study(num_seq, shots, workers, seed=99001) -> dict:
    seqs = E.clifford_sequences(num_seq, max(DEPTHS), SEQ_SEED)
    combos = [("routed", "iqm_garnet"), ("alt4", "iqm_garnet"),
              ("par5", "iqm_garnet"), ("clean_checks", "iqm_garnet"),
              ("clean_checks", "iqm_garnet_native23")]
    tasks = [(v, si, seqs[si], DEPTHS, shots, seed, dev, {})
             for v, dev in combos for si in range(num_seq)]
    t0 = time.perf_counter()
    parts = _run_pool(tasks, workers)
    out = {"num_seq": num_seq, "shots": shots, "seed": seed, "depths": DEPTHS,
           "seconds": time.perf_counter() - t0, "variants": {}}
    for v, pv in parts.items():
        fits, _ = _fits_from_parts(pv, DEPTHS, ("mld5", "const0_mld", "const0",
                                                "lrbd_unif_1_final"))
        out["variants"][v] = fits
    return out


def reset_study(num_seq, shots, workers, seed=99002) -> dict:
    depths = [0, 2, 4, 6, 10, 14]
    seqs = E.clifford_sequences(num_seq, max(depths), SEQ_SEED)
    configs = {
        "reset_tr0_e3pct": {"t_reset_ns": 0.0},
        "reset_tr0_e0": {"t_reset_ns": 0.0, "e_reset": 0.0},
        "noreset": {"builder_kwargs": {"reset": False}},
    }
    tasks = []
    for name, extra in configs.items():
        for si in range(num_seq):
            tasks.append(("routed", si, seqs[si], depths, shots, seed,
                          "iqm_garnet", dict(extra, tag=name)))
    from concurrent.futures import ProcessPoolExecutor, as_completed
    parts: dict = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(_mld_task, t): t[7]["tag"] for t in tasks}
        for fut in as_completed(futs):
            _, si, res, _ = fut.result()
            parts.setdefault(futs[fut], {})[si] = res
    out = {"num_seq": num_seq, "shots": shots, "depths": depths, "configs": {}}
    for name, pv in parts.items():
        fits, _ = _fits_from_parts(pv, depths, ("mld5",))
        out["configs"][name] = fits["mld5"]
    # J's own latency sweep: paired bootstrap of f(no reset) - f(tau_R = 0)
    sw = ROOT / "studies/correction-decoders/qubit/results/J/sweeps/latency"
    if (sw / "iqm_noreset.npz").exists():
        a = E.PointResult.from_npz(str(sw / "iqm_noreset.npz"))
        b = E.PointResult.from_npz(str(sw / "iqm_tr0.npz"))
        Fa = a.tables["mld5"].fidelity()
        Fb = b.tables["mld5"].fidelity()
        d = a.depths
        rng = np.random.default_rng(1)
        diffs = []
        for _ in range(1000):
            idx = rng.integers(0, Fa.shape[0], Fa.shape[0])
            diffs.append(weighted_fit(d, Fa[idx], n_boot=1)["f"]
                         - weighted_fit(d, Fb[idx], n_boot=1)["f"])
        out["J_latency_sweep_paired"] = {
            "f_noreset": weighted_fit(d, Fa, n_boot=1)["f"],
            "f_tr0": weighted_fit(d, Fb, n_boot=1)["f"],
            "diff_mean": float(np.mean(diffs)), "diff_sd": float(np.std(diffs))}
        summ = json.loads((sw / "summary.json").read_text())
        f = {k: v["fits"]["mld5"]["f"] for k, v in summ["configs"].items()}
        out["J_latency_f"] = f
    return out


# ---------------------------------------------------------------------------
# Claim 15: DEM learning spot check
# ---------------------------------------------------------------------------


def _decomposed_graph_dem(dem: stim.DetectorErrorModel) -> stim.DetectorErrorModel:
    """Graph approximation of a hyperedge DEM: every mechanism is replaced by
    its stim graphlike decomposition, each component kept as an independent
    edge with the mechanism's probability (what a matching decoder uses)."""
    lines = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        comp = []
        for t in inst.targets_copy():
            if t.is_separator():
                lines.append((p, comp))
                comp = []
            else:
                comp.append(t)
        lines.append((p, comp))
    text = []
    for p, comp in lines:
        parts = []
        for t in comp:
            parts.append(f"D{t.val}" if t.is_relative_detector_id() else f"L{t.val}")
        text.append(f"error({p}) " + " ".join(parts))
    text.append(f"detector D{dem.num_detectors - 1}")
    return stim.DetectorErrorModel("\n".join(text))


def dem_spot_check(workers: int) -> dict:
    import dem_learning as D
    import hardware_model as H
    res_dir = ROOT / "studies/correction-decoders/qubit/results/J/dem_learning"
    out = {}
    cases = [("iqm_garnet", 0.0, 2, 1), ("iqm_garnet_het0.5", 0.5, 0, 0)]
    for tag, sigma, depth, si in cases:
        js = json.loads((res_dir / f"{tag}.json").read_text())
        row = next(r for r in js["rows"] if r["depth"] == depth and r["seq"] == si)
        nominal = H.DEVICES["iqm_garnet"]
        truth = H.heterogeneous(nominal, sigma, 11) if sigma > 0 else nominal
        seqs = E.clifford_sequences(js["num_seq"], max(js["depths"]), H.SEQ_SEED)
        t0 = time.perf_counter()
        mine = D.study_circuit((truth, nominal, seqs[si], depth, si,
                                js["n_train"], js["n_test"],
                                777 + 1000 * si + depth,
                                (0.1, 0.25, 0.5, 2.0, 4.0, 10.0)))
        cmp = {}
        for name, v in row["decoders"].items():
            cmp[name] = (v["errors"], mine["decoders"][name]["errors"])
        out[f"{tag}_m{depth}_s{si}"] = {
            "errors_J_vs_rerun": cmp,
            "all_equal": all(a == b for a, b in cmp.values()),
            "seconds": time.perf_counter() - t0}
    # Independent estimator and fairer graph baseline on one circuit
    seqs = E.clifford_sequences(4, 22, H.SEQ_SEED)
    lc = H.build_hw_lrb_circuit(seqs[0], 2, H.IQM_GARNET)
    dem = lc.circuit.detector_error_model()
    slots = M.lrb_detector_slots(lc)
    sampler = lc.circuit.compile_detector_sampler(seed=4242)
    train, _ = sampler.sample(1_000_000, separate_observables=True)
    test, obs = lc.circuit.compile_detector_sampler(seed=4243).sample(
        200_000, separate_observables=True)
    flips = obs[:, 0]
    # my own p_ij (Chen et al. Eq. S11, written from the paper's form)
    x = train.astype(np.float64)
    mean = x.mean(0)
    tab = D.footprint_table(dem)
    pairs = sorted({(fp[a], fp[b]) for fp in tab if len(fp) > 1
                    for a in range(len(fp)) for b in range(a + 1, len(fp))})
    true_pairs = D.true_pij(tab, pairs)
    rel = []
    for i, j in pairs:
        xij = (x[:, i] * x[:, j]).mean()
        num = xij - mean[i] * mean[j]
        den = 1 - 2 * mean[i] - 2 * mean[j] + 4 * xij
        pij = 0.5 - 0.5 * math.sqrt(max(1 - 4 * num / den, 0))
        if true_pairs[(i, j)] > 1e-3:
            rel.append(abs(pij - true_pairs[(i, j)]) / true_pairs[(i, j)])
    out["my_pij_median_rel_err_1e6"] = float(np.median(rel))
    out["my_pij_pairs_ptrue_gt_1e-3"] = len(rel)

    def err(dm):
        llr = M.TransferMatrixMLD(dm, slots).log_likelihood_ratio(test)
        return float(np.mean((llr < 0) != flips))
    e_or = err(dem)
    out["graph_baselines_m2_seq0"] = {"oracle_eps": e_or}
    try:
        dec = lc.circuit.detector_error_model(decompose_errors=True,
                                              ignore_decomposition_failures=True)
        gdem = _decomposed_graph_dem(dec)
        out["graph_baselines_m2_seq0"]["decomposed_remaining_hyperedges"] = sum(
            1 for inst in gdem.flattened() if inst.type == "error" and sum(
                1 for t in inst.targets_copy() if t.is_relative_detector_id()) > 2)
        e_g = err(gdem)
        out["graph_baselines_m2_seq0"]["decomposed_true_graph_rel"] = (
            e_g - e_or) / e_or
    except Exception as exc:
        out["graph_baselines_m2_seq0"]["decomposed_true_graph_rel"] = str(exc)
    # J's pairs-only construction on the same data for comparison
    nom_tab = D.footprint_table(dem)
    cands = [fp for fp in nom_tab if fp]
    mom = D.ParityMoments(train)
    graph = D.estimate_footprints(mom, [fp for fp in cands if len(fp) <= 2])
    e_j = err(D.dem_from_table(D.learned_table(graph, nom_tab), dem.num_detectors))
    out["graph_baselines_m2_seq0"]["J_pairs_only_rel"] = (e_j - e_or) / e_or
    # true DEM with every footprint of size > 2 dropped (no re-weighting)
    trunc = {fp: e for fp, e in nom_tab.items() if len(fp) <= 2}
    e_t = err(D.dem_from_table(trunc, dem.num_detectors))
    out["graph_baselines_m2_seq0"]["true_dem_hyperedges_dropped_rel"] = (
        e_t - e_or) / e_or
    sizes = {}
    for fp, e in nom_tab.items():
        sizes[len(fp)] = sizes.get(len(fp), 0.0) + D.total_probability(e)
    out["footprint_weight_by_size_m2_seq0"] = sizes
    # aggregate checks straight from J's JSON
    agg = {}
    for tag in ("iqm_garnet", "iqm_t100", "ion_helios", "iqm_garnet_het0.5",
                "iqm_t100_het0.5"):
        js = json.loads((res_dir / f"{tag}.json").read_text())
        for depth, e in js["per_depth"].items():
            dct = e["decoders"]
            agg[f"{tag}/m{depth}"] = {k: round(100 * dct[k]["rel_diff"], 2)
                                      for k in ("learned_100000",
                                                "learned_1000000",
                                                "learned_1000000_oracle_split",
                                                "graph_1000000", "scaled_0.5",
                                                "scaled_2", "scaled_10",
                                                "nominal")}
    out["J_json_rel_diff_percent"] = agg
    return out


def fit_check() -> dict:
    """Depth-range sensitivity of J's fitted f (J's own per-sequence data):
    points at the |F| noise floor (F ~ 1/sqrt(shots)) bias f upward."""
    base = ROOT / "studies/correction-decoders/qubit/results/J"
    out = {}
    for tag in ("hw_iqm_garnet", "hw_iqm_native23", "hw_ion_helios"):
        P = E.PointResult.from_npz(str(base / tag / "point.npz"))
        d = np.array(P.depths)
        entry = {}
        for method in ("mld5", "const0_mld", "const0"):
            if method not in P.tables:
                continue
            F = P.tables[method].fidelity()
            entry[method] = {f"depths<={dm}": weighted_fit(d[d <= dm], F[:, d <= dm],
                                                           n_boot=200)["f"]
                             for dm in (22, 14, 10, 6)}
            entry[method]["F_mean"] = np.nanmean(F, 0).tolist()
        out[tag] = entry
    # analytic compiled physical RB: 5/6 H pulses per Clifford, 5/3 per step
    p1 = 1.5 * 9e-4
    out["physical_rb_compiled_analytic_f"] = (1 - 4 * p1 / 3) ** (5 / 3)
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("graph")
    g.add_argument("--garnet-file", type=Path, default=None)
    sub.add_parser("idle")
    cf = sub.add_parser("checkfree")
    cf.add_argument("--num-seq", type=int, default=30)
    sub.add_parser("gadget")
    ds = sub.add_parser("distance")
    ds.add_argument("--num-seq", type=int, default=10)
    ml = sub.add_parser("mld")
    ml.add_argument("--num-seq", type=int, default=30)
    ml.add_argument("--shots", type=int, default=20000)
    ml.add_argument("--workers", type=int, default=16)
    rs = sub.add_parser("reset")
    rs.add_argument("--num-seq", type=int, default=10)
    rs.add_argument("--shots", type=int, default=20000)
    rs.add_argument("--workers", type=int, default=16)
    sub.add_parser("fitcheck")
    dm = sub.add_parser("dem")
    dm.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    t0 = time.perf_counter()
    if args.cmd == "graph":
        res = graph_checks(args.garnet_file)
    elif args.cmd == "idle":
        res = idle_checks()
    elif args.cmd == "checkfree":
        res = checkfree_study(args.num_seq)
    elif args.cmd == "gadget":
        res = gadget_checks()
    elif args.cmd == "distance":
        res = distance_study(args.num_seq)
    elif args.cmd == "mld":
        res = mld_study(args.num_seq, args.shots, args.workers)
    elif args.cmd == "reset":
        res = reset_study(args.num_seq, args.shots, args.workers)
    elif args.cmd == "dem":
        res = dem_spot_check(args.workers)
    elif args.cmd == "fitcheck":
        res = fit_check()
    res["wall_seconds"] = time.perf_counter() - t0
    path = _dump(f"{args.cmd}.json", res)
    print(json.dumps(res, indent=1, default=float)[:20000])
    print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
