"""Work package J: a hardware protocol for qubit [[5,1,2]] LRB.

Superconducting (IQM-style square-lattice transmons) first, trapped ions and
neutral atoms second. Everything here is built on the lead's model in
``src/lrb_qubit`` (read-only): circuits are returned as
``lrb_qubit.circuits.LRBCircuit`` objects, so the exact transfer-matrix MLD
(``lrb_qubit.mld``) and the LRB-D policies (``lrb_qubit.baseline``) apply
unchanged.

Sections
--------
1. Layout. The 12 check couplings (the Tanner graph) are exactly the 3x3 grid
   graph, so on a square lattice the embedding is a 3x3 patch, unique up to
   lattice symmetry. The logical coupling (2,3) closes 5-cycles, so it cannot be
   native on any bipartite lattice; in the patch, data 2 and 3 sit at opposite
   corners (distance 4). ``GARNET_COUPLERS`` is IQM Garnet's 30-coupler map
   (copied from IQM's qiskit-on-iqm fake backend, see the notes) and
   ``garnet_patches`` lists the 3x3 patches it contains.
2. Compilation (IQM native gates: PRX pulses, virtual RZ, CZ):
   CNOT = H.CZ.H; one check round = 12 CZ in 6 layers with one simultaneous
   (multiplexed) readout of the four ancillas; logical S, S-dagger, Z are
   virtual phases; logical X, Z are tracked in the Pauli frame; SWAP(2,3) inside
   logical H is a relabelling (frame bit), so H_L is one layer of five pulses;
   CZ(2,3) is routed with 8 CZ through ancillas a7, a5 (both in |0> between
   rounds) and a bridge on data 0 (``routed_cz23_layers``).
3. Device model. Published parameters are converted to Pauli channels:
   depolarizing strength from average infidelity, p = r (d+1)/d, and a
   Pauli-twirl of T1/T2 decay for idling (Tomita and Svore 2014, Eqs. 9-10).
   ``DEVICES`` holds the parameter sets; ``si1000_equivalent`` maps a device to
   effective SI1000 scale factors for the lead's builder.
4. Circuit builder ``build_hw_lrb_circuit`` (layered, with idle noise in every
   layer from the layer duration) mirroring ``lrb_qubit.circuits.build_lrb_circuit``.
5. Runner and analysis: ``run_hw_point`` mirrors ``lrb_qubit.experiment.run_point``
   (LRB-D policies, exact MLD, const0, gap histograms) on the hardware circuits;
   ``decay_fit`` and ``gap_table`` produce the reported numbers.

CLI (writes JSON/NPZ under studies/correction-decoders/qubit/results/J/):

    python studies/correction-decoders/qubit/scripts/hardware_model.py layout
    python studies/correction-decoders/qubit/scripts/hardware_model.py counts
    python studies/correction-decoders/qubit/scripts/hardware_model.py mapping
    python studies/correction-decoders/qubit/scripts/hardware_model.py simulate --device iqm_garnet --tag main --workers 16
    python studies/correction-decoders/qubit/scripts/hardware_model.py analyze --tag main
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

if __name__ == "__main__":
    # One BLAS thread per process (see API.md): re-exec before numpy is used.
    _THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    if any(os.environ.get(_v) != "1" for _v in _THREAD_VARS):
        for _v in _THREAD_VARS:
            os.environ[_v] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import stim

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import code as C  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

RESULTS = ROOT / "studies" / "correction-decoders" / "qubit" / "results" / "J"
DATA = C.DATA_QUBITS
ANCILLAS = C.ANCILLA_QUBITS

# ---------------------------------------------------------------------------
# 1. Layout
# ---------------------------------------------------------------------------


def stabilizer_support(s: int) -> tuple[int, ...]:
    """Code (data) qubits in the support of stabilizer ``s``."""
    return tuple(q for q, ch in enumerate(C.STABILIZER_PAULIS[s]) if ch != "_")


def tanner_edges() -> list[tuple[int, int]]:
    """The 12 (data, ancilla) couplings used by the check circuits."""
    return sorted((q, C.STABILIZER_ANCILLA[s]) for s in range(4)
                  for q in stabilizer_support(s))


LOGICAL_GATE_EDGES = ((2, 3),)  # SWAP(2,3) in logical H, CZ(2,3) in logical S

# The unique square-lattice embedding of the Tanner graph (row, column).
PATCH_COORDS = {0: (0, 0), 7: (0, 1), 2: (0, 2),
                5: (1, 0), 1: (1, 1), 6: (1, 2),
                3: (2, 0), 8: (2, 1), 4: (2, 2)}

# IQM Garnet couplers, verbatim from IQM's qiskit-on-iqm fake backend
# (src/iqm/qiskit_iqm/fake_backends/fake_garnet.py, ``qubit_connectivity``).
GARNET_COUPLERS = (
    ("QB1", "QB2"), ("QB1", "QB4"), ("QB2", "QB5"), ("QB3", "QB4"),
    ("QB8", "QB3"), ("QB4", "QB5"), ("QB9", "QB4"), ("QB5", "QB6"),
    ("QB10", "QB5"), ("QB6", "QB7"), ("QB11", "QB6"), ("QB12", "QB7"),
    ("QB8", "QB9"), ("QB8", "QB13"), ("QB9", "QB10"), ("QB9", "QB14"),
    ("QB10", "QB11"), ("QB10", "QB15"), ("QB11", "QB12"), ("QB16", "QB11"),
    ("QB17", "QB12"), ("QB13", "QB14"), ("QB14", "QB15"), ("QB18", "QB14"),
    ("QB16", "QB15"), ("QB19", "QB15"), ("QB16", "QB17"), ("QB16", "QB20"),
    ("QB18", "QB19"), ("QB19", "QB20"))

# Lattice coordinates (row, column) reconstructed from the coupler list; every
# coupler joins lattice nearest neighbours (checked in the tests).
GARNET_COORDS = {
    "QB1": (0, 1), "QB2": (0, 2),
    "QB3": (1, 0), "QB4": (1, 1), "QB5": (1, 2), "QB6": (1, 3), "QB7": (1, 4),
    "QB8": (2, 0), "QB9": (2, 1), "QB10": (2, 2), "QB11": (2, 3), "QB12": (2, 4),
    "QB13": (3, 0), "QB14": (3, 1), "QB15": (3, 2), "QB16": (3, 3),
    "QB17": (3, 4),
    "QB18": (4, 1), "QB19": (4, 2), "QB20": (4, 3)}


def _graph(edges):
    import networkx as nx
    graph = nx.Graph()
    graph.add_edges_from(edges)
    return graph


def embeddings(host_edges, pattern_edges) -> list[dict]:
    """All subgraph monomorphisms pattern -> host, as {pattern_node: host_node}."""
    from networkx.algorithms import isomorphism
    host, pattern = _graph(host_edges), _graph(pattern_edges)
    matcher = isomorphism.GraphMatcher(host, pattern)
    return [{v: k for k, v in m.items()}
            for m in matcher.subgraph_monomorphisms_iter()]


def square_grid_edges(rows: int, cols: int) -> list:
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append(((r, c), (r, c + 1)))
            if r + 1 < rows:
                edges.append(((r, c), (r + 1, c)))
    return edges


def shortest_cycle_through(edges, edge) -> int:
    """Length of the shortest cycle of ``edges`` that uses ``edge``."""
    import networkx as nx
    graph = _graph([e for e in edges if set(e) != set(edge)])
    return nx.shortest_path_length(graph, edge[0], edge[1]) + 1


def garnet_patches() -> list[dict]:
    """3x3 patches of Garnet: one representative labelling per distinct qubit set.

    Each entry maps code qubits 0-8 to Garnet qubit names and records the
    data-2/data-3 coupler distance on the full Garnet graph.
    """
    import networkx as nx
    garnet = _graph(GARNET_COUPLERS)
    found: dict[frozenset, dict] = {}
    count = 0
    for emb in embeddings(GARNET_COUPLERS, tanner_edges()):
        count += 1
        key = frozenset(emb.values())
        if key not in found:
            found[key] = {
                "labels": {int(k): v for k, v in sorted(emb.items())},
                "distance_q2_q3": nx.shortest_path_length(garnet, emb[2], emb[3]),
            }
    patches = list(found.values())
    for patch in patches:
        patch["labellings_per_patch"] = count // max(len(found), 1)
    return patches


def ascii_patch() -> str:
    return "\n".join([
        "  q0 ---- a7 ---- q2        (row 0)",
        "  |       |       |",
        "  a5 ---- q1 ---- a6        (row 1)",
        "  |       |       |",
        "  q3 ---- a8 ---- q4        (row 2)",
        "",
        "  a5: Z0 Z1 Z3   a6: Z1 Z2 Z4   a7: X0 X1 X2   a8: X1 X3 X4  (frame 0)",
        "  fold axis (main diagonal): q0, q1, q4; logical coupling q2..q3 at",
        "  Manhattan distance 4 (opposite corners)."])


def ascii_garnet(patch_labels: Optional[dict] = None) -> str:
    """Garnet coupling lattice with an optional patch overlay."""
    inverse = {v: k for k, v in (patch_labels or {}).items()}
    rows = 1 + max(r for r, _ in GARNET_COORDS.values())
    cols = 1 + max(c for _, c in GARNET_COORDS.values())
    grid = [["      " for _ in range(cols)] for _ in range(rows)]
    for name, (r, c) in GARNET_COORDS.items():
        tag = name
        if name in inverse:
            k = inverse[name]
            tag = f"{name}={'q' if k < 5 else 'a'}{k}"
        grid[r][c] = f"{tag:<10}"
    return "\n".join("".join(cell if cell.strip() else " " * 10 for cell in row)
                     for row in grid)


# ---------------------------------------------------------------------------
# 2. Compilation to CZ + single-qubit pulses
# ---------------------------------------------------------------------------

# Frame f: code data qubit -> physical data qubit (SWAP(2,3) is a relabelling)
PERM = {0: (0, 1, 2, 3, 4), 1: (0, 1, 3, 2, 4)}
# Frame f: physical ancilla that measures code stabilizer (S_Z1, S_Z2, S_X1, S_X2)
ANC_OF_STAB = {0: (5, 6, 7, 8), 1: (7, 8, 5, 6)}

VIRTUAL_GATES = {"S", "S_DAG", "Z"}


@dataclass(frozen=True)
class Layer:
    """One time step. ``kind``: 1q, 2q, meas, reset, virtual (zero time,
    noiseless: virtual-Z phases and Pauli-frame Paulis), release (marks
    ancillas returned to |0>, no operation)."""

    kind: str
    ops: tuple

    @property
    def qubits(self) -> set:
        return {q for _, t in self.ops for q in t}


def _cnot_layers(control: int, target: int) -> list[Layer]:
    """CNOT = H(t) CZ(c, t) H(t)."""
    return [Layer("1q", (("H", (target,)),)),
            Layer("2q", (("CZ", (control, target)),)),
            Layer("1q", (("H", (target,)),))]


def routed_cz23_layers(bridge: int = 0) -> list[Layer]:
    """CZ between physical data 2 and 3 on the 3x3 patch: 8 CZ.

    Copy Z2 into ancilla a7 and Z3 into a5 (both |0>), apply CZ(a7, a5) through
    the bridge data qubit (bridge = 0: CNOT(a7->q0) CZ(q0,a5) CNOT(a7->q0)
    CZ(q0,a5) leaves q0 unchanged and applies (-1)^{z2 z3}), then uncopy.
    ``bridge = 4`` uses ancillas a6, a8 and data 4 instead.
    """
    if bridge == 0:
        a2, a3 = 7, 5
    elif bridge == 4:
        a2, a3 = 6, 8
    else:
        raise ValueError("bridge must be 0 or 4")
    copy = [Layer("1q", (("H", (a2,)), ("H", (a3,)))),
            Layer("2q", (("CZ", (2, a2)), ("CZ", (3, a3)))),
            Layer("1q", (("H", (a2,)), ("H", (a3,))))]
    mid = (_cnot_layers(a2, bridge) + [Layer("2q", (("CZ", (bridge, a3)),))]
           + _cnot_layers(a2, bridge) + [Layer("2q", (("CZ", (bridge, a3)),))])
    # If an ancilla was not reset it holds its last outcome m; the routed
    # gate then also applies Z3^m(a2) Z2^m(a3). These known byproducts are
    # Pauli-frame updates ("feedback" layer: zero time, noiseless; a no-op
    # when the ancilla was reset).
    frame_fix = [Layer("feedback", (("Z", (a2, 3)), ("Z", (a3, 2))))]
    release = [Layer("release", (("I", (a2,)), ("I", (a3,))))]
    return compact(copy + mid + copy + frame_fix + release)


def native_cz23_layers() -> list[Layer]:
    return [Layer("2q", (("CZ", (2, 3)),))]


def cz23_layers(mode: str, bridge: int = 0) -> list[Layer]:
    if mode == "routed":
        return routed_cz23_layers(bridge)
    if mode == "native":
        return native_cz23_layers()
    raise ValueError(f"unknown CZ(2,3) mode {mode!r}")


def swap23_layers(mode: str, bridge: int = 0) -> list[Layer]:
    """Physical SWAP(2,3) as three CNOTs built from CZ(2,3) (for counting)."""
    layers: list[Layer] = []
    for c, t in ((2, 3), (3, 2), (2, 3)):
        layers += [Layer("1q", (("H", (t,)),))] + cz23_layers(mode, bridge) + [
            Layer("1q", (("H", (t,)),))]
    return compact(layers)


def compact(layers: list[Layer]) -> list[Layer]:
    """Merge consecutive single-qubit layers.

    Two adjacent 1q layers are merged when every qubit they share carries H
    in both (the H.H pairs cancel); otherwise they stay separate.
    """
    out: list[Layer] = []
    for layer in layers:
        if out and layer.kind == "1q" and out[-1].kind == "1q":
            prev = list(out[-1].ops)
            index = {t: i for i, (_, t) in enumerate(prev)}
            overlap = [op for op in layer.ops if op[1] in index]
            if all(op[0] == "H" and prev[index[op[1]]][0] == "H"
                   for op in overlap):
                cancel = {op[1] for op in overlap}
                merged = ([o for o in prev if o[1] not in cancel]
                          + [o for o in layer.ops if o[1] not in cancel])
                if merged:
                    out[-1] = Layer("1q", tuple(merged))
                else:
                    out.pop()
                continue
        out.append(layer)
    return out


def check_round_layers(frame: int, reset: bool = True) -> list[Layer]:
    """One CZ-native check round in frame ``frame``.

    Z checks first (3 CZ layers, ancillas H-conjugated), then all data are
    H-rotated for the X checks (3 CZ layers). The four ancillas are read out in
    ONE simultaneous (multiplexed) measurement layer, ordered by code
    stabilizer, then reset (forward rounds) unless ``reset`` is False.
    """
    perm, anc = PERM[frame], ANC_OF_STAB[frame]
    supports = [tuple(perm[q] for q in stabilizer_support(s)) for s in range(4)]

    def schedule(sa, sb):
        # Both checks contain data 1: put it in different layers.
        a = [1] + [q for q in sa if q != 1]
        b = [q for q in sb if q != 1]
        b = [b[0], 1, b[1]]
        return [((0, a[k]), (1, b[k])) for k in range(3)]

    layers = [Layer("1q", (("H", (anc[0],)), ("H", (anc[1],))))]
    for lay in schedule(supports[0], supports[1]):
        layers.append(Layer("2q", tuple(("CZ", (d, anc[i])) for i, d in lay)))
    layers.append(Layer("1q", tuple(("H", (q,)) for q in DATA)
                        + tuple(("H", (a,)) for a in anc)))
    for lay in schedule(supports[2], supports[3]):
        layers.append(Layer("2q", tuple(("CZ", (anc[2 + i], d))
                                        for i, d in lay)))
    layers.append(Layer("1q", tuple(("H", (q,)) for q in DATA)
                        + (("H", (anc[2],)), ("H", (anc[3],)))))
    layers.append(Layer("meas", tuple(("M", (anc[s],)) for s in range(4))))
    if reset:
        layers.append(Layer("reset", tuple(("R", (a,)) for a in anc)))
    return layers


def compile_logical_ops(code_ops: Sequence[C.Operation], frame: int, *,
                        cz23: str = "routed", bridge: int = 0,
                        virtual_paulis: bool = True,
                        virtual_swap: bool = True) -> tuple[list[Layer], int]:
    """Physical layers for code-level logical-gate operations.

    Returns the layers and the frame after them. S, S_DAG and Z are virtual
    (zero-duration frame changes on the drive, McKay et al. 2017); X is a
    Pauli-frame update when ``virtual_paulis`` (else one pi pulse); H is one
    pi/2 pulse plus virtual Z; SWAP(2,3) toggles the frame.
    """
    layers: list[Layer] = []
    pending: list = []
    kind: Optional[str] = None

    def flush():
        nonlocal pending, kind
        if pending:
            layers.append(Layer(kind, tuple(pending)))
        pending, kind = [], None

    for gate, targets in code_ops:
        perm = PERM[frame]
        if gate == "SWAP":
            if virtual_swap:
                frame ^= 1
            else:
                flush()
                layers += swap23_layers(cz23, bridge)
            continue
        if gate == "CZ":
            flush()
            layers += cz23_layers(cz23, bridge)
            continue
        if gate == "I":
            continue
        q = perm[targets[0]]
        op_kind = ("virtual" if gate in VIRTUAL_GATES
                   or (gate == "X" and virtual_paulis) else "1q")
        if gate not in VIRTUAL_GATES | {"H", "X"}:
            raise ValueError(f"unexpected logical-gate op {gate}")
        if kind != op_kind or any(q in t for _, t in pending):
            flush()
            kind = op_kind
        pending.append((gate, (q,)))
    flush()
    return compact(layers), frame


def layer_counts(layers: Iterable[Layer]) -> Counter:
    """Gate and layer counts (CZ, 1q pulses, virtual ops, layers per kind)."""
    counts: Counter = Counter()
    for layer in layers:
        counts[f"layers_{layer.kind}"] += 1
        if layer.kind == "2q":
            counts["cz"] += len(layer.ops)
        elif layer.kind == "1q":
            counts["pulses_1q"] += len(layer.ops)
        elif layer.kind == "virtual":
            counts["virtual"] += len(layer.ops)
        elif layer.kind == "meas":
            counts["measurements"] += len(layer.ops)
        elif layer.kind == "reset":
            counts["resets"] += len(layer.ops)
    return counts


def layers_duration_ns(layers: Iterable[Layer], device: "Device") -> float:
    return sum(device.duration_ns(layer.kind) for layer in layers)


# ---------------------------------------------------------------------------
# 3. Device model
# ---------------------------------------------------------------------------


def depolarizing_from_infidelity(r: float, num_qubits: int) -> float:
    """Depolarizing probability with average gate infidelity ``r``.

    A d-dimensional depolarizing channel with total Pauli-error probability p
    has process fidelity 1 - p and average fidelity 1 - p d/(d+1) (Nielsen
    2002), so p = r (d+1)/d: 3r/2 for one qubit, 5r/4 for two.
    """
    d = 2 ** num_qubits
    return r * (d + 1) / d


def pauli_twirl_idle(t_ns: float, t1_us: float, t2_us: float
                     ) -> tuple[float, float, float]:
    """Pauli-twirl of amplitude and phase damping (Tomita and Svore 2014).

    p_X = p_Y = (1 - exp(-t/T1))/4, p_Z = (1 - exp(-t/T2))/2 - (1 - exp(-t/T1))/4.
    """
    t = t_ns / 1000.0
    a = 0.0 if math.isinf(t1_us) else 1.0 - math.exp(-t / t1_us)
    b = 0.0 if math.isinf(t2_us) else 1.0 - math.exp(-t / t2_us)
    px = a / 4.0
    pz = max(b / 2.0 - a / 4.0, 0.0)
    return px, px, pz


@dataclass(frozen=True)
class Device:
    """Hardware parameters (times in ns, coherence in microseconds).

    Error rates are average gate infidelities (as reported by RB/IRB) and
    assignment errors; they are converted to Pauli channels when circuits are
    built. ``extra_idle`` adds a depolarizing error per layer of a given kind
    to every active qubit not acted on (e.g. measurement crosstalk or a
    per-layer memory error), as ((kind, p), ...). ``overrides`` sets
    per-physical-qubit values as ((qubit, field, value), ...).
    """

    name: str
    t1_us: float = math.inf
    t2_us: float = math.inf
    t_1q_ns: float = 20.0
    t_2q_ns: float = 40.0
    t_meas_ns: float = 1000.0
    t_reset_ns: float = 500.0
    r_1q: float = 0.0
    r_2q: float = 0.0
    e_meas: float = 0.0
    e_reset: float = 0.0
    extra_idle: tuple = ()
    cz23: str = "routed"
    overrides: tuple = ()
    source: str = ""

    def value(self, name: str, qubit: Optional[int] = None) -> float:
        if qubit is not None:
            for q, key, val in self.overrides:
                if q == qubit and key == name:
                    return val
        return getattr(self, name)

    def duration_ns(self, kind: str) -> float:
        return {"1q": self.t_1q_ns, "2q": self.t_2q_ns,
                "meas": self.t_meas_ns, "reset": self.t_reset_ns}.get(kind, 0.0)

    def p1(self, q: int) -> float:
        return depolarizing_from_infidelity(self.value("r_1q", q), 1)

    def p2(self, a: int, b: int) -> float:
        r = max(self.value("r_2q", a), self.value("r_2q", b))
        return depolarizing_from_infidelity(r, 2)

    def idle(self, q: int, t_ns: float) -> tuple[float, float, float]:
        return pauli_twirl_idle(t_ns, self.value("t1_us", q),
                                self.value("t2_us", q))

    def extra(self, kind: str) -> float:
        return sum(p for k, p in self.extra_idle if k == kind)

    def scaled_errors(self, factor: float) -> "Device":
        """Same device with every error probability multiplied by ``factor``
        (coherence-limited idling scaled through 1/T1, 1/T2)."""
        return replace(
            self, name=f"{self.name}_x{factor:g}",
            t1_us=self.t1_us / factor, t2_us=self.t2_us / factor,
            r_1q=self.r_1q * factor, r_2q=self.r_2q * factor,
            e_meas=self.e_meas * factor, e_reset=self.e_reset * factor,
            extra_idle=tuple((k, p * factor) for k, p in self.extra_idle),
            overrides=tuple((q, k, (v / factor if k in ("t1_us", "t2_us")
                                    else v * factor))
                            for q, k, v in self.overrides))


IQM_GARNET = Device(
    name="iqm_garnet",
    t1_us=40.0,        # Abdurakhimov et al. 2024: "T1 from the order of 40 us"
    t2_us=21.05,       # arXiv:2512.23009 Table 3: Garnet median T2 (echo)
    t_1q_ns=20.0,      # IQM fake Garnet backend: prx 20 ns
    t_2q_ns=40.0,      # fake backend cz 40 ns; whitepaper: CZ 20-40 ns
    t_meas_ns=1000.0,  # 800 ns readout pulse (Renger et al. 2025, Tab. S4)
                       # + 200 ns integration/ring-down (assumed)
    t_reset_ns=500.0,  # feedback latency of measure + cc_prx: UNVERIFIED
    r_1q=9e-4,         # Abdurakhimov et al. 2024: median 1q error (RB)
    r_2q=5e-3,         # Abdurakhimov et al. 2024: median CZ error (IRB)
    e_meas=3e-2,       # Abdurakhimov et al. 2024: median simultaneous readout
    e_reset=3e-2,      # conditional reset inherits the misassignment (assumed)
    cz23="routed",
    source="IQM Garnet medians (arXiv:2408.12433) + T2e snapshot "
           "(arXiv:2512.23009) + IQM readout pulse (arXiv:2503.10903)")

DEVICES: dict[str, Device] = {
    "iqm_garnet": IQM_GARNET,
    # Same gates, no dynamical decoupling: Ramsey T2* (arXiv:2512.23009).
    "iqm_garnet_ramsey": replace(IQM_GARNET, name="iqm_garnet_ramsey",
                                 t2_us=8.43),
    # Calibration snapshot medians of arXiv:2512.23009 (T1 27.42 us,
    # PRX 99.86 %, CZ 99.40 %, readout 97.29 %).
    "iqm_garnet_snapshot": replace(IQM_GARNET, name="iqm_garnet_snapshot",
                                   t1_us=27.42, r_1q=1.4e-3, r_2q=6.0e-3,
                                   e_meas=2.71e-2, e_reset=2.71e-2),
    # IQM's stated T1 target (>100 us) with T2 = T1 (projection).
    "iqm_t100": replace(IQM_GARNET, name="iqm_t100", t1_us=100.0,
                        t2_us=100.0),
    # Hypothetical long-range coupler: native CZ(2,3).
    "iqm_garnet_native23": replace(IQM_GARNET, name="iqm_garnet_native23",
                                   cz23="native"),
    # Quantinuum Helios (arXiv:2511.05465): 1q 2.5e-5, 2q 7.9e-4, SPAM
    # 4.8e-4, linear memory error 5e-4 per depth-1 layer (applied to every
    # idle qubit per two-qubit layer), MCMR global crosstalk 4.8e-5 per
    # measured ion (x4 ancillas per readout layer). All-to-all: native CZ(2,3).
    "ion_helios": Device(
        name="ion_helios", t_1q_ns=1e4, t_2q_ns=7e4, t_meas_ns=1e5,
        t_reset_ns=0.0, r_1q=2.5e-5, r_2q=7.9e-4, e_meas=4.8e-4,
        e_reset=0.0, extra_idle=(("2q", 5e-4), ("meas", 4 * 4.8e-5),
                                 ("meas", 5e-4)),
        cz23="native",
        source="Ransford et al. 2025, arXiv:2511.05465 (abstract, Sec. III)"),
    # Harvard/QuEra Rb array (arXiv:2312.03982, arXiv:2304.05420): CZ 99.5 %,
    # local 1q 99.912 %, readout 99.8 % with 500 us imaging that costs data
    # <~0.1 %, T2 ~ 1.5 s effective, ~200 us moves per gate layer. Native
    # CZ(2,3) after a move.
    "atom_rb": Device(
        name="atom_rb", t1_us=math.inf, t2_us=1.5e6, t_1q_ns=1e3,
        t_2q_ns=2e5, t_meas_ns=5e5, t_reset_ns=0.0, r_1q=8.8e-4,
        r_2q=5e-3, e_meas=2e-3, e_reset=2e-3, extra_idle=(("meas", 1e-3),),
        cz23="native",
        source="Bluvstein et al. 2024 (arXiv:2312.03982); Evered et al. 2023 "
               "(arXiv:2304.05420)"),
}


def heterogeneous(device: Device, sigma: float, seed: int,
                  fields: Sequence[str] = ("t1_us", "t2_us", "e_meas",
                                           "r_1q", "r_2q")) -> Device:
    """Per-qubit log-normal spread of the given fields (median preserved)."""
    rng = np.random.default_rng(seed)
    overrides = []
    for q in range(9):
        for key in fields:
            factor = float(np.exp(sigma * rng.standard_normal()))
            overrides.append((q, key, getattr(device, key) * factor))
    return replace(device, name=f"{device.name}_het{sigma:g}_s{seed}",
                   overrides=tuple(overrides))


def idle_probability(device: Device, t_ns: float, q: int = 0) -> float:
    return sum(device.idle(q, t_ns))


def si1000_equivalent(device: Device) -> dict:
    """Effective SI1000 parameters for a device (homogeneous medians).

    p is the CZ depolarizing strength; every other term is expressed as a
    multiple of p, with its physical origin:

    - two_qubit (p): CZ average infidelity x 5/4;
    - one_qubit (0.1 p in SI1000): single-pulse infidelity x 3/2;
    - meas_idle (2 p, "ResonatorIdle"): Pauli-twirled T1/T2 decay of a data
      qubit during one readout window, plus extra measurement-layer terms;
    - meas_flip (5 p): readout assignment error;
    - meas_depol (p in the repo): no counterpart (dephasing of the measured
      ancilla is irrelevant; its T1 decay is part of the assignment error);
    - reset_idle (2 p): decay of a data qubit during the reset/feedback window;
    - reset_flip (2 p): residual excitation after (conditional) reset;
    - gate_idle (Idle p/10 in SI1000, omitted by the repo): decay during one
      CZ layer.
    """
    p = depolarizing_from_infidelity(device.r_2q, 2)
    meas_idle = idle_probability(device, device.t_meas_ns) + device.extra("meas")
    reset_idle = idle_probability(device, device.t_reset_ns) + device.extra(
        "reset")
    gate_idle = idle_probability(device, device.t_2q_ns) + device.extra("2q")
    one = depolarizing_from_infidelity(device.r_1q, 1)
    factors = {
        "one_qubit": one / p, "two_qubit": 1.0, "meas_idle": meas_idle / p,
        "meas_depol": 0.0, "meas_flip": device.e_meas / p,
        "reset_idle": reset_idle / p, "reset_flip": device.e_reset / p}
    return {"p": p, "factors": factors, "gate_idle_over_p": gate_idle / p,
            "absolute": {"cz": p, "one_qubit": one, "meas_idle": meas_idle,
                         "meas_flip": device.e_meas, "reset_idle": reset_idle,
                         "reset_flip": device.e_reset, "gate_idle_cz_layer":
                         gate_idle},
            "si1000_reference": {"one_qubit": 0.1, "two_qubit": 1.0,
                                 "meas_idle": 2.0, "meas_flip": 5.0,
                                 "reset_idle": 2.0, "reset_flip": 2.0,
                                 "gate_idle": 0.1}}


def si1000_noise_for(device: Device) -> tuple[float, Q.SI1000]:
    """(p, SI1000 scale factors) for the lead's ``build_lrb_circuit``."""
    eq = si1000_equivalent(device)
    return eq["p"], Q.SI1000(**eq["factors"])


# ---------------------------------------------------------------------------
# 4. Hardware LRB circuit builder
# ---------------------------------------------------------------------------


class _Emitter:
    """Appends layers to a Stim circuit with gate, idle and SPAM noise."""

    def __init__(self, device: Optional[Device], noisy: bool = True):
        self.circuit = stim.Circuit()
        self.device = device
        self.noisy = bool(noisy and device is not None)
        self.active: set = set(DATA)  # qubits holding quantum information
        self.time_ns = 0.0
        self.counts: Counter = Counter()
        # record index of an ancilla's last outcome while it is not reset
        self.unreset: dict[int, int] = {}

    def _idle(self, qubits: Iterable[int], kind: str) -> None:
        if not self.noisy:
            return
        dev = self.device
        t_ns = dev.duration_ns(kind)
        extra = dev.extra(kind)
        for q in sorted(qubits):
            px, py, pz = dev.idle(q, t_ns)
            if px + py + pz > 0:
                self.circuit.append("PAULI_CHANNEL_1", [q], [px, py, pz])
            if extra > 0:
                self.circuit.append("DEPOLARIZE1", [q], extra)

    def emit(self, layer: Layer) -> list[int]:
        circuit, dev = self.circuit, self.device
        records: list[int] = []
        if layer.kind == "virtual":
            for gate, targets in layer.ops:
                circuit.append(gate, list(targets))
            self.counts["virtual"] += len(layer.ops)
            return records
        if layer.kind == "release":
            self.active -= layer.qubits
            return records
        if layer.kind == "feedback":
            # Pauli-frame update conditioned on an unreset ancilla's outcome.
            for gate, (ancilla, target) in layer.ops:
                if ancilla in self.unreset:
                    rec = self.unreset[ancilla] - circuit.num_measurements
                    circuit.append("C" + gate, [stim.target_rec(rec), target])
            return records
        acted = layer.qubits
        if layer.kind in ("1q", "2q"):
            for gate, targets in layer.ops:
                circuit.append(gate, list(targets))
                if self.noisy:
                    if layer.kind == "1q":
                        circuit.append("DEPOLARIZE1", list(targets),
                                       dev.p1(targets[0]))
                    else:
                        circuit.append("DEPOLARIZE2", list(targets),
                                       dev.p2(*targets))
            self.active |= acted
            self.counts["cz" if layer.kind == "2q" else "pulses_1q"] += len(
                layer.ops)
        elif layer.kind == "meas":
            for _, (q,) in layer.ops:
                if self.noisy and dev.value("e_meas", q) > 0:
                    circuit.append("X_ERROR", [q], dev.value("e_meas", q))
                records.append(circuit.num_measurements)
                self.unreset[q] = circuit.num_measurements
                circuit.append("M", [q])
            self.counts["measurements"] += len(layer.ops)
        elif layer.kind == "reset":
            for _, (q,) in layer.ops:
                self.unreset.pop(q, None)
                circuit.append("R", [q])
                if self.noisy and dev.value("e_reset", q) > 0:
                    circuit.append("X_ERROR", [q], dev.value("e_reset", q))
            self.counts["resets"] += len(layer.ops)
        else:
            raise ValueError(f"unknown layer kind {layer.kind}")
        self._idle(self.active - acted, layer.kind)
        if layer.kind in ("meas", "reset"):
            self.active -= acted
        if dev is not None:
            self.time_ns += dev.duration_ns(layer.kind)
        self.counts[f"layers_{layer.kind}"] += 1
        return records

    def emit_all(self, layers: Iterable[Layer]) -> list[int]:
        records: list[int] = []
        for layer in layers:
            records += self.emit(layer)
        return records


def _back_subsets(segment, stabilizers) -> tuple[tuple[int, ...], ...]:
    inverse = C.data_tableau(segment).inverse()
    return tuple(C.decompose_in_stabilizers(inverse(C.stabilizer_pauli(s)))
                 for s in stabilizers)


def _xor(lists: Iterable[Iterable[int]]) -> list[int]:
    acc: Counter = Counter()
    for lst in lists:
        acc.update(lst)
    return sorted(k for k, v in acc.items() if v % 2)


@dataclass
class BuildInfo:
    counts: Counter
    duration_ns: float
    frames: list
    round_start_ns: list = field(default_factory=list)


def build_hw_lrb_circuit(cliffords: Sequence[str], depth: int,
                         device: Optional[Device], *,
                         terminal: str = "data5",
                         terminal_noise: bool = False,
                         forward_check_rounds: Optional[Iterable[int]] = None,
                         terminal_check: bool = True,
                         reset: bool = True,
                         virtual_paulis: bool = True,
                         virtual_swap: bool = True,
                         cz23: Optional[str] = None,
                         bridge: int = 0,
                         noisy: bool = True,
                         return_info: bool = False):
    """Hardware-compiled LRB circuit of depth ``depth`` (cf. build_lrb_circuit).

    Same protocol as the repo (ideal encoding, forward Clifford + check round,
    mirror inverses without checks, terminal check, ideal terminal X readout)
    but compiled to CZ + pulses with simultaneous ancilla readout, virtual
    SWAP(2,3)/Z/Paulis, routed or native CZ(2,3), and T1/T2 idle noise in
    every layer. ``reset=False`` drops the ancilla resets and uses the XOR of
    consecutive outcomes of the same physical ancilla as the raw syndrome.
    Detectors and the observable have the repo's meaning, so ``rounds``,
    ``detector_keys`` and ``raw_syndromes`` behave as for ``LRBCircuit``.
    """
    if terminal not in ("data2", "data5"):
        raise ValueError("terminal must be 'data2' or 'data5'.")
    forward = list(cliffords[:depth])
    if len(forward) < depth:
        raise ValueError("Clifford sequence is shorter than the depth.")
    forward_set = (set(range(depth)) if forward_check_rounds is None
                   else {int(r) for r in forward_check_rounds})
    mode = cz23 or (device.cz23 if device is not None else "routed")

    em = _Emitter(device, noisy)
    for gate, targets in C.encoding_operations():
        em.circuit.append(gate, list(targets))  # ideal encoding (frame 0)

    frame = 0
    frames: list[int] = []
    rounds: list[int] = []
    raw_records: dict[int, list[list[int]]] = {}
    last_meas: dict[int, int] = {}
    segments: list[list[C.Operation]] = []
    pending: list[C.Operation] = []
    round_start: list[float] = []

    def run_round(label: int, do_reset: bool) -> None:
        round_start.append(em.time_ns)
        layers = check_round_layers(frame, reset=do_reset and reset)
        records = em.emit_all(layers)
        anc = ANC_OF_STAB[frame]
        raw = []
        for s in range(4):
            lst = [records[s]]
            if not reset and anc[s] in last_meas:
                lst.append(last_meas[anc[s]])
            raw.append(lst)
        for s in range(4):
            last_meas[anc[s]] = records[s]
        rounds.append(label)
        frames.append(frame)
        raw_records[label] = raw
        segments.append(list(pending))
        pending.clear()

    def logical(descriptor: str, inverse: bool) -> None:
        nonlocal frame
        ops = C.logical_clifford_operations(descriptor, inverse=inverse)
        layers, frame = compile_logical_ops(
            ops, frame, cz23=mode, bridge=bridge,
            virtual_paulis=virtual_paulis, virtual_swap=virtual_swap)
        em.emit_all(layers)
        pending.extend(ops)

    for j, descriptor in enumerate(forward):
        logical(descriptor, False)
        if j in forward_set:
            run_round(j, True)
    for descriptor in reversed(forward):
        logical(descriptor, True)
    if terminal_check:
        run_round(depth, False)
    segments.append(list(pending))
    pending.clear()

    perm = PERM[frame]
    read_code = (0, 3) if terminal == "data2" else DATA
    for q in read_code:
        em.circuit.append("H", [perm[q]])
    data_record = {}
    for q in read_code:
        if terminal_noise and em.noisy and device.value("e_meas", perm[q]) > 0:
            em.circuit.append("X_ERROR", [perm[q]],
                              device.value("e_meas", perm[q]))
        data_record[q] = em.circuit.num_measurements
        em.circuit.append("M", [perm[q]])

    circuit = em.circuit
    detector_keys: list[tuple[int, int]] = []
    back: dict[int, tuple[tuple[int, ...], ...]] = {}
    previous: Optional[list[list[int]]] = None
    for position, label in enumerate(rounds):
        raw = raw_records[label]
        subsets = (tuple(() for _ in range(4)) if previous is None
                   else _back_subsets(segments[position], range(4)))
        back[label] = subsets
        for s in range(4):
            targets = _xor([raw[s]] + ([previous[k] for k in subsets[s]]
                                       if previous is not None else []))
            circuit.append("DETECTOR", [stim.target_rec(r - circuit.num_measurements)
                                        for r in targets], [s, label])
            detector_keys.append((label, s))
        previous = raw
    if terminal == "data5":
        subsets = _back_subsets(segments[-1], C.X_STABILIZERS)
        for s, subset in zip(C.X_STABILIZERS, subsets):
            lists = [[data_record[q]] for q in stabilizer_support(s)]
            if previous is not None:
                lists += [previous[k] for k in subset]
            targets = _xor(lists)
            circuit.append("DETECTOR", [stim.target_rec(r - circuit.num_measurements)
                                        for r in targets], [s, depth + 1])
            detector_keys.append((depth + 1, s))
    circuit.append("OBSERVABLE_INCLUDE",
                   [stim.target_rec(data_record[q] - circuit.num_measurements)
                    for q in (0, 3)], 0)
    lc = Q.LRBCircuit(circuit=circuit, depth=depth, rounds=rounds,
                      detector_keys=detector_keys, back=back,
                      terminal=terminal, segments=segments)
    if return_info:
        return lc, BuildInfo(em.counts, em.time_ns, frames, round_start)
    return lc


def build_hw_const0_circuit(cliffords, depth, device, **kwargs):
    """No check rounds: ideal transversal X readout of all data (const=0)."""
    return build_hw_lrb_circuit(cliffords, depth, device, terminal="data5",
                                forward_check_rounds=(), terminal_check=False,
                                **kwargs)


# ---------------------------------------------------------------------------
# Gate counts and durations
# ---------------------------------------------------------------------------


def logical_token_counts(cz23: str = "routed", bridge: int = 0,
                         virtual_swap: bool = True,
                         virtual_paulis: bool = True) -> dict:
    """CZ / pulse / layer counts of each logical gate token (frame 0)."""
    out = {}
    for token in ("H", "P", "X", "Z", "I"):
        layers, _ = compile_logical_ops(
            C.logical_gate_operations(token), 0, cz23=cz23, bridge=bridge,
            virtual_paulis=virtual_paulis, virtual_swap=virtual_swap)
        out[token] = dict(layer_counts(layers))
    return out


def clifford_catalog_counts(device: Device, cz23: Optional[str] = None,
                            virtual_swap: bool = True,
                            virtual_paulis: bool = True) -> dict:
    """Per-Clifford CZ count and duration, averaged over the 24 descriptors."""
    mode = cz23 or device.cz23
    rows = []
    for descriptor in C.SINGLE_QUBIT_CLIFFORDS:
        layers, frame = compile_logical_ops(
            C.logical_clifford_operations(descriptor), 0, cz23=mode,
            virtual_swap=virtual_swap, virtual_paulis=virtual_paulis)
        counts = layer_counts(layers)
        rows.append({"descriptor": descriptor, "cz": counts["cz"],
                     "pulses_1q": counts["pulses_1q"],
                     "duration_ns": layers_duration_ns(layers, device),
                     "frame_toggle": frame})
    cz = np.array([r["cz"] for r in rows], float)
    dur = np.array([r["duration_ns"] for r in rows], float)
    pulses = np.array([r["pulses_1q"] for r in rows], float)
    return {"mean_cz": cz.mean(), "mean_pulses_1q": pulses.mean(),
            "mean_duration_ns": dur.mean(), "max_cz": cz.max(),
            "max_duration_ns": dur.max(), "rows": rows}


def repo_gate_counts() -> dict:
    """Two-qubit gate counts of the repo's (lead's) gate model, for reference."""
    two = []
    for descriptor in C.SINGLE_QUBIT_CLIFFORDS:
        ops = C.logical_clifford_operations(descriptor)
        two.append(sum(1 for g, _ in ops if g in ("CX", "CZ", "SWAP")))
    round_two = sum(1 for g, _ in C.check_round_operations() if g == "CX")
    return {"mean_two_qubit_per_clifford": float(np.mean(two)),
            "two_qubit_per_round": round_two}


def gate_count_report(device: Device = IQM_GARNET) -> dict:
    round_layers = check_round_layers(0, reset=True)
    terminal_layers = check_round_layers(0, reset=False)
    report = {
        "device": device.name,
        "check_round": dict(layer_counts(round_layers)),
        "check_round_duration_ns": layers_duration_ns(round_layers, device),
        "terminal_round_duration_ns": layers_duration_ns(terminal_layers,
                                                         device),
        "check_round_gate_time_ns": layers_duration_ns(
            [l for l in round_layers if l.kind in ("1q", "2q")], device),
        "tokens_routed": logical_token_counts("routed"),
        "tokens_native": logical_token_counts("native"),
        "tokens_physical_swap_routed": logical_token_counts(
            "routed", virtual_swap=False),
        "clifford_routed": {k: v for k, v in clifford_catalog_counts(
            device, "routed").items() if k != "rows"},
        "clifford_native": {k: v for k, v in clifford_catalog_counts(
            device, "native").items() if k != "rows"},
        "clifford_routed_physical_swap": {
            k: v for k, v in clifford_catalog_counts(
                device, "routed", virtual_swap=False).items() if k != "rows"},
        "repo_gate_model": repo_gate_counts(),
    }
    cliff = report["clifford_routed"]
    report["depth_unit_routed"] = {
        "description": "one forward Clifford + its inverse + one check round",
        "cz": 2 * cliff["mean_cz"] + report["check_round"]["cz"],
        "duration_ns": 2 * cliff["mean_duration_ns"]
        + report["check_round_duration_ns"]}
    return report


# ---------------------------------------------------------------------------
# 5. Runner (mirrors lrb_qubit.experiment.run_point)
# ---------------------------------------------------------------------------


def run_hw_point(device: Device, sequences: Sequence[Sequence[str]],
                 depths: Sequence[int], shots: int, *,
                 decode_shots: Optional[int] = None,
                 const_checks: Sequence[int] = tuple(range(0, 23)),
                 unif_checks: Sequence[int] = tuple(range(1, 23)),
                 final_variant_policies=(("unif", 1), ("const", 1)),
                 batch_size: int = 100_000, seed: int = 12345,
                 builder_kwargs: Optional[dict] = None,
                 label_p: Optional[float] = None,
                 check_every: Optional[int] = None,
                 with_mld2: bool = True) -> E.PointResult:
    """All LRB-D policies, exact MLD (mld5/mld2) and const0 on shared shots.

    With ``check_every = k`` only the forward rounds j*k - 1 (the rounds that
    ``unif=k`` uses) are physically executed; LRB-D is then reported as
    ``lrbd_all`` (all executed rounds clean) and ``lrbd_all_final``.
    """
    kwargs = dict(builder_kwargs or {})
    depths = list(depths)
    p_label = (si1000_equivalent(device)["p"] if label_p is None else label_p)
    result = E.PointResult(p_label, depths, len(sequences))
    policies = ([] if check_every else E.policy_names(const_checks,
                                                      unif_checks))
    if check_every:
        policies = [("lrbd_all", "all", 0)]
        final_variant_policies = (("all", 0),)
    decode_shots = shots if decode_shots is None else min(decode_shots, shots)
    for seq_index, sequence in enumerate(sequences):
        for d_index, depth in enumerate(depths):
            if check_every:
                kwargs["forward_check_rounds"] = [
                    j * check_every - 1
                    for j in range(1, depth // check_every + 1)]
            lc5 = build_hw_lrb_circuit(sequence, depth, device,
                                       terminal="data5", **kwargs)
            mld5 = M.TransferMatrixMLD(lc5.circuit.detector_error_model(),
                                       M.lrb_detector_slots(lc5))
            decoders = [("mld5", mld5, slice(None))]
            if with_mld2:
                lc2 = build_hw_lrb_circuit(sequence, depth, device,
                                           terminal="data2", **kwargs)
                mld2 = M.TransferMatrixMLD(lc2.circuit.detector_error_model(),
                                           M.lrb_detector_slots(lc2))
                decoders.append(("mld2", mld2, slice(0, 4 * lc5.num_rounds)))
            n_round = 4 * lc5.num_rounds
            sampler = lc5.circuit.compile_detector_sampler(
                seed=seed + 1000 * seq_index + d_index)
            done = 0
            while done < shots:
                n = min(batch_size, shots - done)
                dets, obs = sampler.sample(n, separate_observables=True)
                flip = obs[:, 0]
                raw = lc5.raw_syndromes(dets)
                final_clean = ~dets[:, n_round:].any(axis=1)
                for name, policy, k in policies:
                    used = (lc5.rounds if policy == "all"
                            else B.used_rounds(policy, k, depth))
                    acc = B.accept_mask(raw, lc5.rounds, used)
                    result.table(name).add(seq_index, d_index, n,
                                           int(acc.sum()),
                                           int((acc & flip).sum()))
                    if (policy, k) in final_variant_policies:
                        acc_f = acc & final_clean
                        result.table(name + "_final").add(
                            seq_index, d_index, n, int(acc_f.sum()),
                            int((acc_f & flip).sum()))
                n_dec = min(n, max(decode_shots - done, 0))
                if n_dec > 0:
                    for name, decoder, cols in decoders:
                        llr = decoder.log_likelihood_ratio(dets[:n_dec, cols])
                        wrong = (llr < 0) != flip[:n_dec]
                        result.table(name).add(seq_index, d_index, n_dec,
                                               n_dec, int(wrong.sum()))
                        E._histogram_gap(result.hist(name)[seq_index, d_index],
                                         np.abs(llr), wrong)
                done += n
            if 0 in const_checks:
                lc0 = build_hw_const0_circuit(
                    sequence, depth, device,
                    **{k: v for k, v in kwargs.items()
                       if k not in ("reset", "forward_check_rounds")})
                dec0 = M.TransferMatrixMLD(lc0.circuit.detector_error_model(),
                                           M.lrb_detector_slots(lc0))
                sampler0 = lc0.circuit.compile_detector_sampler(
                    seed=seed + 7 + 1000 * seq_index + d_index)
                done = 0
                while done < decode_shots:
                    n = min(batch_size, decode_shots - done)
                    dets, obs = sampler0.sample(n, separate_observables=True)
                    flip = obs[:, 0]
                    acc = ~dets.any(axis=1)
                    result.table("const0").add(seq_index, d_index, n,
                                               int(acc.sum()),
                                               int((acc & flip).sum()))
                    llr = dec0.log_likelihood_ratio(dets)
                    wrong = (llr < 0) != flip
                    result.table("const0_mld").add(seq_index, d_index, n, n,
                                                   int(wrong.sum()))
                    E._histogram_gap(result.hist("const0_mld")[seq_index,
                                                               d_index],
                                     np.abs(llr), wrong)
                    done += n
    return result


def physical_rb_hw(device: Device, sequences, depths, shots: int,
                   virtual_paulis: bool = True, seed: int = 999,
                   idle_ns_per_clifford: float = 0.0) -> B.CountTable:
    """Matched physical RB on one qubit with IQM-style compilation.

    H and (if not virtual) X are pi/2 or pi pulses with depolarizing
    3 r_1q / 2; S, S-dagger, Z are virtual. ``idle_ns_per_clifford`` adds
    T1/T2 idling per Clifford (e.g. the logical circuit's wall-clock time).
    """
    table = B.CountTable.zeros(len(sequences), len(depths))
    p1 = depolarizing_from_infidelity(device.r_1q, 1)
    idle = device.idle(0, idle_ns_per_clifford)
    for si, sequence in enumerate(sequences):
        for di, depth in enumerate(depths):
            circuit = stim.Circuit()
            circuit.append("I", [0])
            tokens = []
            for descriptor in sequence[:depth]:
                tokens.append(C.clifford_gate_sequence(descriptor))
            for descriptor in reversed(sequence[:depth]):
                tokens.append(C.clifford_inverse_sequence(descriptor))
            for word in tokens:
                for token in word:
                    gate = {"H": "H", "H_INV": "H", "P": "S", "P_INV": "S_DAG",
                            "X": "X", "X_INV": "X", "Z": "Z", "Z_INV": "Z",
                            "I": None}[token]
                    if gate is None:
                        continue
                    circuit.append(gate, [0])
                    if gate == "H" or (gate == "X" and not virtual_paulis):
                        circuit.append("DEPOLARIZE1", [0], p1)
                if sum(idle) > 0:
                    circuit.append("PAULI_CHANNEL_1", [0], list(idle))
            circuit.append("M", [0])
            circuit.append("OBSERVABLE_INCLUDE", [stim.target_rec(-1)], 0)
            sampler = circuit.compile_detector_sampler(seed=seed + 1000 * si + di)
            _, obs = sampler.sample(shots, separate_observables=True)
            table.add(si, di, shots, shots, int(obs[:, 0].sum()))
    return table


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


def _fit_log_linear(depths, fid, sem):
    depths = np.asarray(depths, float)
    fid = np.asarray(fid, float)
    sem = np.asarray(sem, float)
    ok = np.isfinite(fid) & (fid > 0) & (fid < 1) & np.isfinite(sem)
    if ok.sum() < 2:
        return np.nan, np.nan
    y = np.log(fid[ok])
    w = (fid[ok] / np.maximum(sem[ok], 1e-6)) ** 2  # 1/var of log F
    x = depths[ok]
    W = w.sum()
    xm, ym = (w * x).sum() / W, (w * y).sum() / W
    slope = (w * (x - xm) * (y - ym)).sum() / (w * (x - xm) ** 2).sum()
    return float(np.exp(ym - slope * xm)), float(np.exp(slope))


def decay_fit(table: B.CountTable, depths: Sequence[int], n_boot: int = 400,
              seed: int = 7, min_depth: int = 0,
              min_accepted: int = 1000) -> dict:
    """Fit F(m) = b f^m to the sequence-averaged F (offset fixed at 0).

    Weighted least squares on log F (weights from the standard error of the
    mean over sequences); uncertainties from a bootstrap over sequences.
    Depths whose pooled accepted shots fall below ``min_accepted`` are
    dropped (postselected methods only). r = (1 - f)/2 is the logical error
    per depth step (one Clifford, its inverse and one check round), d = 2.
    """
    fid = table.fidelity()  # (seq, depth)
    depths = np.asarray(depths)
    keep = (depths >= min_depth) & (table.accepted.sum(0) >= min_accepted)
    fid = fid[:, keep]
    use_depths = depths[keep]

    def stats(sample):
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(sample, axis=0)
            sem = np.nanstd(sample, axis=0) / np.sqrt(
                np.maximum(np.sum(np.isfinite(sample), axis=0), 1))
        # binomial floor for the standard error
        return mean, np.maximum(sem, 1e-4)

    mean, sem = stats(fid)
    b, f = _fit_log_linear(use_depths, mean, sem)
    rng = np.random.default_rng(seed)
    boots = []
    n = fid.shape[0]
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bm, bs = stats(fid[idx])
        boots.append(_fit_log_linear(use_depths, bm, bs))
    boots = np.array(boots, float)
    good = np.all(np.isfinite(boots), axis=1)
    b_err = float(np.std(boots[good, 0])) if good.any() else np.nan
    f_err = float(np.std(boots[good, 1])) if good.any() else np.nan
    return {"b": b, "b_err": b_err, "f": f, "f_err": f_err,
            "r_per_clifford": (1 - f) / 2, "r_err": f_err / 2,
            "depths": use_depths.tolist(), "F_mean": mean.tolist(),
            "F_sem": sem.tolist()}


def table_summary(table: B.CountTable) -> dict:
    fid = table.fidelity()
    rej = table.rejection()
    n = fid.shape[0]
    with np.errstate(invalid="ignore"):
        return {"F_mean": np.nanmean(fid, 0).tolist(),
                "F_sem": (np.nanstd(fid, 0) / np.sqrt(n)).tolist(),
                "rejection_mean": rej.mean(0).tolist(),
                "rejection_sem": (rej.std(0) / np.sqrt(n)).tolist(),
                "accepted_total": table.accepted.sum(0).tolist(),
                "shots_total": table.shots.sum(0).tolist()}


def gap_table(hist: np.ndarray, target_acceptance: Sequence[float]) -> dict:
    """Gap postselection: per depth, accept shots with |LLR| >= threshold.

    ``hist`` has shape (seq, depth, bins, 2) (correct, wrong). For each target
    acceptance the threshold is the largest bin edge whose pooled acceptance
    is at least the target; F is computed per sequence and averaged.
    """
    edges = E.GAP_EDGES
    seqs, depths = hist.shape[:2]
    out = {}
    for target in target_acceptance:
        F_mean, F_sem, acc_mean, thresholds = [], [], [], []
        for d in range(depths):
            pooled = hist[:, d].sum(axis=0)  # bins x 2
            tail = pooled[::-1].sum(axis=1).cumsum()[::-1]  # shots with bin >= k
            total = tail[0]
            k = int(np.max(np.nonzero(tail >= target * total)[0])) if total else 0
            per_seq_acc = hist[:, d, k:].sum(axis=(1, 2))
            per_seq_err = hist[:, d, k:, 1].sum(axis=1)
            per_seq_shots = hist[:, d].sum(axis=(1, 2))
            with np.errstate(invalid="ignore", divide="ignore"):
                fid = np.abs(1 - 2 * per_seq_err / per_seq_acc)
            F_mean.append(float(np.nanmean(fid)))
            F_sem.append(float(np.nanstd(fid) / np.sqrt(seqs)))
            acc_mean.append(float(np.mean(per_seq_acc / np.maximum(
                per_seq_shots, 1))))
            thresholds.append(float(edges[k]))
        out[f"{target:g}"] = {"F_mean": F_mean, "F_sem": F_sem,
                              "acceptance": acc_mean,
                              "threshold": thresholds}
    return out


def gap_matched_to(hist: np.ndarray, table: B.CountTable) -> dict:
    """Gap postselection at the acceptance of another method (per depth)."""
    acc = table.accepted.sum(0) / np.maximum(table.shots.sum(0), 1)
    F_mean, F_sem, got = [], [], []
    for d in range(hist.shape[1]):
        pooled = hist[:, d].sum(axis=0)
        tail = pooled[::-1].sum(axis=1).cumsum()[::-1]
        total = tail[0]
        ok = np.nonzero(tail >= acc[d] * total)[0]
        k = int(ok.max()) if len(ok) else 0
        per_acc = hist[:, d, k:].sum(axis=(1, 2))
        per_err = hist[:, d, k:, 1].sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            fid = np.abs(1 - 2 * per_err / per_acc)
        F_mean.append(float(np.nanmean(fid)))
        F_sem.append(float(np.nanstd(fid) / np.sqrt(hist.shape[0])))
        got.append(float(tail[k] / max(total, 1)))
    return {"F_mean": F_mean, "F_sem": F_sem, "acceptance": got,
            "target_acceptance": acc.tolist()}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
SEQ_SEED = 20261028  # same sequences as results/grid_main
MAIN_METHODS = ("mld5", "mld2", "const0_mld", "const0", "lrbd_const_1",
                "lrbd_const_1_final", "lrbd_unif_1", "lrbd_unif_1_final",
                "lrbd_unif_2", "lrbd_unif_5")


REDUCED_CONST = (0, 1, 2, 3, 5, 10, 22)
REDUCED_UNIF = (1, 2, 3, 5, 11, 22)


def _task(args):
    (kind, device, seq_index, sequence, depths, shots, decode_shots, seed, kw,
     run_kw) = args
    start = time.perf_counter()
    if kind == "hw":
        res = run_hw_point(device, [sequence], depths, shots,
                           decode_shots=decode_shots,
                           seed=seed + 1_000_000 * seq_index,
                           builder_kwargs=kw, **run_kw)
    else:  # repo builder with mapped SI1000 factors
        p, noise = si1000_noise_for(device)
        extra = {k: v for k, v in run_kw.items()
                 if k in ("const_checks", "unif_checks")}
        res = E.run_point(p, [sequence], depths, shots,
                          decode_shots=decode_shots,
                          seed=seed + 1_000_000 * seq_index, noise=noise,
                          **extra)
    return seq_index, res, time.perf_counter() - start


def _merge(parts: dict, p: float, depths, n_seq) -> E.PointResult:
    merged = E.PointResult(p, list(depths), n_seq)
    for si, part in parts.items():
        for name, table in part.tables.items():
            t = merged.table(name)
            t.shots[si] = table.shots[0]
            t.accepted[si] = table.accepted[0]
            t.errors[si] = table.errors[0]
        for name, hist in part.gap_hist.items():
            merged.hist(name)[si] = hist[0]
    return merged


def simulate(device: Device, tag: str, num_seq: int, depths, shots: int,
             decode_shots: Optional[int], workers: int, seed: int,
             kind: str = "hw", builder_kwargs: Optional[dict] = None,
             run_kwargs: Optional[dict] = None) -> Path:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    out = RESULTS / tag
    out.mkdir(parents=True, exist_ok=True)
    sequences = E.clifford_sequences(num_seq, max(depths), SEQ_SEED)
    meta = {"device": asdict(device), "kind": kind, "num_seq": num_seq,
            "depths": list(depths), "shots": shots,
            "decode_shots": decode_shots, "seq_seed": SEQ_SEED, "seed": seed,
            "builder_kwargs": builder_kwargs or {},
            "run_kwargs": run_kwargs or {},
            "si1000_equivalent": si1000_equivalent(device)}
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=str))
    tasks = [(kind, device, si, sequences[si], list(depths), shots,
              decode_shots, seed, builder_kwargs or {}, run_kwargs or {})
             for si in range(num_seq)]
    parts = {}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_task, t) for t in tasks]
        for count, fut in enumerate(as_completed(futures), 1):
            si, res, sec = fut.result()
            parts[si] = res
            print(f"[{time.perf_counter() - start:7.1f}s] {tag}: seq {si} "
                  f"done ({sec:.1f}s), {count}/{len(tasks)}", flush=True)
    p_label = si1000_equivalent(device)["p"]
    merged = _merge(parts, p_label, depths, num_seq)
    merged.to_npz(str(out / "point.npz"))
    rb = physical_rb_hw(device, sequences, depths, min(shots, 100_000))
    np.savez_compressed(out / "rb.npz", shots=rb.shots, accepted=rb.accepted,
                        errors=rb.errors)
    return out


def _sweep_task(args):
    name, device, kw, every, si, sequence, depths, shots, seed = args
    start = time.perf_counter()
    res = run_hw_point(device, [sequence], depths, shots,
                       seed=seed + 1_000_000 * si, builder_kwargs=kw,
                       check_every=every, with_mld2=False,
                       const_checks=(0, 1), unif_checks=(1,))
    return name, si, res, time.perf_counter() - start


def sweep_configs(which: str) -> list[tuple]:
    """(name, device, builder kwargs, check_every) for each sweep."""
    configs = []
    if which == "coherence":
        for t in (None, 50.0, 100.0, 200.0, 500.0, 1000.0):
            dev = (IQM_GARNET if t is None else replace(
                IQM_GARNET, name=f"iqm_T{t:g}", t1_us=t, t2_us=t))
            for every in (1, 2, 5):
                configs.append((f"{dev.name}_k{every}", dev, {}, every))
    elif which == "latency":
        for tr in (0.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0):
            dev = replace(IQM_GARNET, name=f"iqm_tr{tr:g}", t_reset_ns=tr)
            configs.append((dev.name, dev, {}, 1))
        configs.append(("iqm_noreset", IQM_GARNET, {"reset": False}, 1))
    elif which == "readout":
        for tm in (250.0, 500.0, 1000.0, 2000.0):
            dev = replace(IQM_GARNET, name=f"iqm_tm{tm:g}", t_meas_ns=tm)
            configs.append((dev.name, dev, {}, 1))
    elif which == "platforms_sparse":
        for key in ("ion_helios", "atom_rb", "iqm_t100", "iqm_garnet_native23"):
            for every in (2, 5):
                configs.append((f"{key}_k{every}", DEVICES[key], {}, every))
    else:
        raise ValueError(which)
    return configs


def run_sweep(which: str, num_seq: int, depths, shots: int, workers: int,
              seed: int = 4242) -> dict:
    from concurrent.futures import ProcessPoolExecutor, as_completed
    configs = sweep_configs(which)
    out = RESULTS / "sweeps" / which
    out.mkdir(parents=True, exist_ok=True)
    sequences = E.clifford_sequences(num_seq, max(depths), SEQ_SEED)
    tasks = [(name, dev, kw, every, si, sequences[si], list(depths), shots,
              seed) for name, dev, kw, every in configs
             for si in range(num_seq)]
    parts: dict = {name: {} for name, *_ in configs}
    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_sweep_task, t) for t in tasks]
        for count, fut in enumerate(as_completed(futures), 1):
            name, si, res, sec = fut.result()
            parts[name][si] = res
            if count % max(1, len(tasks) // 10) == 0:
                print(f"[{time.perf_counter() - start:7.1f}s] {which}: "
                      f"{count}/{len(tasks)}", flush=True)
    summary = {"which": which, "depths": list(depths), "num_seq": num_seq,
               "shots": shots, "configs": {}}
    for name, dev, kw, every in configs:
        merged = _merge(parts[name], si1000_equivalent(dev)["p"], depths,
                        num_seq)
        merged.to_npz(str(out / f"{name}.npz"))
        entry = {"device": asdict(dev), "kwargs": kw, "check_every": every,
                 "fits": {}, "methods": {}}
        for method, table in merged.tables.items():
            entry["methods"][method] = table_summary(table)
            entry["fits"][method] = decay_fit(table, depths)
        summary["configs"][name] = entry
    (out / "summary.json").write_text(json.dumps(summary, indent=1,
                                                 default=str))
    return summary


def physical_rb_baselines(num_seq: int = 30, shots: int = 100_000,
                          depths=DEPTHS) -> dict:
    """The lead's matched physical RB (every gate token followed by
    DEPOLARIZE1(0.1 p_RB)) at p_RB = p and 5p, with p the device's effective
    SI1000 p, plus the IQM-compiled physical RB (pulses only)."""
    sequences = E.clifford_sequences(num_seq, max(depths), SEQ_SEED)
    out = {}
    for name in ("iqm_garnet", "iqm_t100", "ion_helios", "atom_rb"):
        dev = DEVICES[name]
        p = si1000_equivalent(dev)["p"]
        entry = {"p": p}
        for label, p_rb in (("p_rb_eq_p", p), ("p_rb_eq_5p", 5 * p)):
            table = E.run_physical_rb(p_rb, sequences, depths, shots)
            entry[label] = {"p_rb": p_rb, **decay_fit(table, depths)}
        table = physical_rb_hw(dev, sequences, depths, shots)
        entry["compiled_pulses"] = decay_fit(table, depths)
        out[name] = entry
    (RESULTS / "physical_rb.json").write_text(json.dumps(out, indent=1))
    return out


def analyze(tag: str) -> dict:
    out = RESULTS / tag
    meta = json.loads((out / "meta.json").read_text())
    point = E.PointResult.from_npz(str(out / "point.npz"))
    depths = point.depths
    summary = {"tag": tag, "device": meta["device"]["name"], "kind": meta["kind"],
               "depths": depths, "num_seq": meta["num_seq"],
               "shots": meta["shots"], "methods": {}, "fits": {}, "gap": {}}
    for name in MAIN_METHODS:
        if name in point.tables:
            summary["methods"][name] = table_summary(point.tables[name])
            summary["fits"][name] = decay_fit(point.tables[name], depths)
    for name in ("mld5", "mld2", "const0_mld"):
        if name in point.gap_hist:
            summary["gap"][name] = gap_table(point.gap_hist[name],
                                             (0.99, 0.9, 0.75, 0.5))
            for ref in ("lrbd_unif_1", "lrbd_const_1", "lrbd_unif_1_final",
                        "const0"):
                if ref in point.tables:
                    summary["gap"][f"{name}_at_{ref}"] = gap_matched_to(
                        point.gap_hist[name], point.tables[ref])
    if (out / "rb.npz").exists():
        rb = np.load(out / "rb.npz")
        table = B.CountTable(rb["shots"], rb["accepted"], rb["errors"])
        summary["fits"]["physical_rb"] = decay_fit(table, depths)
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="WP J hardware model")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("layout")
    sub.add_parser("counts")
    sub.add_parser("mapping")
    sim = sub.add_parser("simulate")
    sim.add_argument("--device", default="iqm_garnet")
    sim.add_argument("--tag", required=True)
    sim.add_argument("--kind", default="hw", choices=("hw", "repo"))
    sim.add_argument("--num-seq", type=int, default=30)
    sim.add_argument("--depths", default=",".join(map(str, DEPTHS)))
    sim.add_argument("--shots", type=int, default=100_000)
    sim.add_argument("--decode-shots", type=int, default=None)
    sim.add_argument("--workers", type=int, default=4)
    sim.add_argument("--seed", type=int, default=2026)
    sim.add_argument("--no-reset", action="store_true")
    sim.add_argument("--physical-x", action="store_true")
    sim.add_argument("--t-reset-ns", type=float, default=None)
    sim.add_argument("--t-meas-ns", type=float, default=None)
    sim.add_argument("--terminal-noise", action="store_true")
    sim.add_argument("--reduced", action="store_true",
                     help="LRB-D policies const 0,1,2,3,5,10,22 and "
                          "unif 1,2,3,5,11,22 only")
    sim.add_argument("--no-mld2", action="store_true")
    ana = sub.add_parser("analyze")
    ana.add_argument("--tag", required=True)
    sub.add_parser("physrb")
    swp = sub.add_parser("sweep")
    swp.add_argument("--which", required=True,
                     choices=("coherence", "latency", "readout",
                              "platforms_sparse"))
    swp.add_argument("--num-seq", type=int, default=10)
    swp.add_argument("--depths", default="0,2,4,6,10,14")
    swp.add_argument("--shots", type=int, default=50_000)
    swp.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    RESULTS.mkdir(parents=True, exist_ok=True)

    if args.cmd == "layout":
        patches = garnet_patches()
        report = {
            "tanner_edges": tanner_edges(),
            "logical_edges": LOGICAL_GATE_EDGES,
            "patch_coords": PATCH_COORDS,
            "grid_embeddings_3x3": len(embeddings(square_grid_edges(3, 3),
                                                  tanner_edges())),
            "grid_embeddings_5x5": len(embeddings(square_grid_edges(5, 5),
                                                  tanner_edges())),
            "shortest_cycle_through_23": shortest_cycle_through(
                tanner_edges() + [(2, 3)], (2, 3)),
            "patch_distance_q2_q3": sum(abs(a - b) for a, b in zip(
                PATCH_COORDS[2], PATCH_COORDS[3])),
            "garnet_patches": patches,
            "ascii_patch": ascii_patch(),
            "ascii_garnet": ascii_garnet(patches[0]["labels"] if patches
                                         else None)}
        (RESULTS / "layout.json").write_text(json.dumps(report, indent=1))
        print(report["ascii_patch"])
        print(report["ascii_garnet"])
        print(json.dumps({k: v for k, v in report.items()
                          if not k.startswith("ascii")}, indent=1))
    elif args.cmd == "counts":
        report = {name: gate_count_report(dev) for name, dev in DEVICES.items()}
        (RESULTS / "gate_counts.json").write_text(json.dumps(report, indent=1,
                                                             default=float))
        print(json.dumps(report["iqm_garnet"], indent=1, default=float))
    elif args.cmd == "mapping":
        report = {name: {"device": asdict(dev), **si1000_equivalent(dev)}
                  for name, dev in DEVICES.items()}
        (RESULTS / "mapping.json").write_text(json.dumps(report, indent=1,
                                                         default=str))
        print(json.dumps({k: {"p": v["p"], "factors": v["factors"]}
                          for k, v in report.items()}, indent=1))
    elif args.cmd == "simulate":
        device = DEVICES[args.device]
        if args.t_reset_ns is not None:
            device = replace(device, t_reset_ns=args.t_reset_ns,
                             name=f"{device.name}_tr{args.t_reset_ns:g}")
        if args.t_meas_ns is not None:
            device = replace(device, t_meas_ns=args.t_meas_ns,
                             name=f"{device.name}_tm{args.t_meas_ns:g}")
        kw = {}
        if args.no_reset:
            kw["reset"] = False
        if args.physical_x:
            kw["virtual_paulis"] = False
        if args.terminal_noise:
            kw["terminal_noise"] = True
        run_kw = {}
        if args.reduced:
            run_kw["const_checks"] = REDUCED_CONST
            run_kw["unif_checks"] = REDUCED_UNIF
        if args.no_mld2 and args.kind == "hw":
            run_kw["with_mld2"] = False
        depths = [int(d) for d in args.depths.split(",")]
        simulate(device, args.tag, args.num_seq, depths, args.shots,
                 args.decode_shots, args.workers, args.seed, args.kind, kw,
                 run_kw)
        summary = analyze(args.tag)
        for name, fit in summary["fits"].items():
            print(f"{name:22s} f = {fit['f']:.4f} +- {fit['f_err']:.4f}  "
                  f"b = {fit['b']:.3f}")
    elif args.cmd == "analyze":
        summary = analyze(args.tag)
        for name, fit in summary["fits"].items():
            print(f"{name:22s} f = {fit['f']:.4f} +- {fit['f_err']:.4f}  "
                  f"b = {fit['b']:.3f}")
    elif args.cmd == "physrb":
        out = physical_rb_baselines()
        for name, entry in out.items():
            print(name, f"p={entry['p']:.3g}",
                  {k: (round(v["f"], 5), round(v["f_err"], 5))
                   for k, v in entry.items() if isinstance(v, dict)})
    elif args.cmd == "sweep":
        depths = [int(d) for d in args.depths.split(",")]
        summary = run_sweep(args.which, args.num_seq, depths, args.shots,
                            args.workers)
        for name, entry in summary["configs"].items():
            line = []
            for method in ("mld5", "const0_mld", "const0", "lrbd_all",
                           "lrbd_unif_1"):
                if method in entry["fits"]:
                    fit = entry["fits"][method]
                    line.append(f"{method} f={fit['f']:.4f}+-{fit['f_err']:.4f}")
            print(f"{name:26s} " + "  ".join(line))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
