"""BP+OSD, BP+LSD, union-find and belief-find decoders (ldpc 2.4.1) for LRB circuits.

Work package C of the qubit ``[[5,1,2]]`` decoder study. The decoders act on the
detector error model (DEM) of an ``LRBCircuit`` (see ``circuits.py``) and
predict the flip of observable 0 (the terminal ``X_L`` readout), so they can be
scored against ``mld.TransferMatrixMLD`` on identical shots.

Check matrices
--------------
``dem_check_matrices(dem, mode)`` turns a DEM into a check matrix ``H``
(detectors x columns), an observable matrix ``L`` (observables x columns) and
column priors. Three column conventions are offered because they matter at
distance 2, where every non-trivial syndrome has two first-order explanations
that differ by a logical operator:

``"mechanism"`` (default)
    One column per distinct (detector set, observable set) of the DEM, i.e.
    one column per Stim error mechanism (exact duplicates are XOR-merged).
    Mechanisms with the same detectors but opposite observable effects stay
    separate columns; a decoder that picks the more probable column picks the
    more probable logical class.
``"dominant"``
    One column per distinct detector set. Its prior is the XOR-combination
    (probability of an odd number of firings) of all mechanisms with that
    set, and its observable is that of the most probable class.
``"ldpc"``
    ``ldpc.ckt_noise.detector_error_model_to_check_matrices`` as used by
    ``ldpc.SinterBpOsdDecoder``. It also keys columns by detector set only,
    but keeps the observable of the *last listed* mechanism, whatever its
    probability. On these circuits about one detector set in four carries
    both observable classes, so this convention decides the logical class of
    those columns by DEM ordering. It is kept to quantify that pitfall.

Mechanisms without detectors (undetectable logical errors, present only with
the two-qubit ``data2`` readout) cannot be represented in ``H``; they are
dropped and their XOR-combined probability is stored in
``CheckMatrices.undetectable_probability``. They shift every decoder equally.

Union-find
----------
ldpc's ``UnionFindDecoder`` has two modes. *Peeling* is the original
Delfosse-Nickerson decoder; it needs a graph (every column of weight 1 or 2,
weight-1 columns are boundary edges) and aborts the interpreter with a C++
exception otherwise. *Matrix solve* (``uf_method`` any non-empty string)
grows clusters on the Tanner graph and solves each cluster by Gaussian
elimination, so it accepts hyperedges. Note the ldpc 2.4.1 API quirks:
``UnionFindDecoder(H)`` with the default ``uf_method=False`` raises
``TypeError``, ``uf_method="peeling"`` silently selects matrix-solve mode
(non-empty strings are truthy), and peeling needs ``uf_method=""``.
ldpc's peeling mode also hangs on some valid high-weight syndromes (p=1e-2,
depth 22), so the standard set uses matrix-solve mode and the pure-Python
``DelfosseNickersonUF`` for textbook peeling.
``graph_check_matrices`` builds the decomposed matching graph from a DEM made
with ``decompose_errors=True``: each graphlike component of a mechanism is an
edge carrying the mechanism's probability, parallel edges are XOR-merged and
keep the observable of their most probable class.
With ``weighted=True`` the decoders pass ``llr = log((1-p)/p)`` and
``bits_per_step=1``: clusters then grow by the most probable (lowest-LLR)
adjacent column first.

All decoders here are deterministic functions of the syndrome, so
``LdpcLrbDecoder.predict`` decodes each distinct syndrome once and broadcasts.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import stim
from scipy.sparse import csc_matrix

MERGE_MODES = ("mechanism", "dominant", "ldpc")


def xor_probability(a: float, b: float) -> float:
    """Probability that exactly one of two independent events occurs."""
    return a + b - 2.0 * a * b


@dataclass
class CheckMatrices:
    """Binary decoding problem extracted from a DEM.

    Attributes:
        check_matrix: ``(num_detectors, num_columns)`` uint8 CSC matrix ``H``.
        observables_matrix: ``(num_observables, num_columns)`` uint8 CSC ``L``.
        priors: ``(num_columns,)`` column probabilities.
        mode: Column convention (``MERGE_MODES`` or ``"graph"``).
        undetectable_probability: XOR-combined probability of dropped
            detector-free mechanisms that flip observable 0.
    """

    check_matrix: csc_matrix
    observables_matrix: csc_matrix
    priors: np.ndarray
    mode: str
    undetectable_probability: float = 0.0

    @property
    def num_detectors(self) -> int:
        return self.check_matrix.shape[0]

    @property
    def num_columns(self) -> int:
        return self.check_matrix.shape[1]

    def llrs(self) -> np.ndarray:
        """Column weights ``log((1 - p) / p)`` (small = probable)."""
        p = np.clip(self.priors, 1e-300, 1.0 - 1e-16)
        return np.log((1.0 - p) / p)

    def observable_flips(self, errors: np.ndarray) -> np.ndarray:
        """``L e mod 2`` for one error vector or a ``(shots, columns)`` batch."""
        errors = np.asarray(errors)
        dense = self.observables_matrix.toarray().astype(np.int64)
        if errors.ndim == 1:
            return (dense @ errors.astype(np.int64)) % 2 == 1
        return (errors.astype(np.int64) @ dense.T) % 2 == 1


def _dem_errors(dem: stim.DetectorErrorModel):
    """Yield ``(probability, components)`` for every error instruction.

    ``components`` is a list of ``(detectors, observables)`` frozensets, one per
    ``^``-separated part (a single part for undecomposed DEMs).
    """
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = instruction.args_copy()[0]
        parts: list[tuple[list[int], list[int]]] = [([], [])]
        for target in instruction.targets_copy():
            if target.is_separator():
                parts.append(([], []))
            elif target.is_relative_detector_id():
                parts[-1][0].append(target.val)
            elif target.is_logical_observable_id():
                parts[-1][1].append(target.val)
        components = []
        for dets, obs in parts:
            # XOR semantics: a target listed twice cancels.
            d = frozenset(x for x in set(dets) if dets.count(x) % 2)
            o = frozenset(x for x in set(obs) if obs.count(x) % 2)
            components.append((d, o))
        yield probability, components


def _xor_sets(components) -> tuple[frozenset, frozenset]:
    dets: frozenset = frozenset()
    obs: frozenset = frozenset()
    for d, o in components:
        dets = dets ^ d
        obs = obs ^ o
    return dets, obs


def _assemble(columns: list[tuple[frozenset, frozenset, float]],
              num_detectors: int, num_observables: int, mode: str,
              undetectable: float) -> CheckMatrices:
    rows, cols, obs_rows, obs_cols = [], [], [], []
    priors = np.zeros(len(columns))
    for j, (dets, obs, prob) in enumerate(columns):
        for d in sorted(dets):
            rows.append(d)
            cols.append(j)
        for o in sorted(obs):
            obs_rows.append(o)
            obs_cols.append(j)
        priors[j] = prob
    h = csc_matrix((np.ones(len(rows), np.uint8), (rows, cols)),
                   shape=(num_detectors, len(columns)), dtype=np.uint8)
    lmat = csc_matrix((np.ones(len(obs_rows), np.uint8), (obs_rows, obs_cols)),
                      shape=(num_observables, len(columns)), dtype=np.uint8)
    return CheckMatrices(h, lmat, priors, mode, undetectable)


def dem_check_matrices(dem: stim.DetectorErrorModel,
                       mode: str = "mechanism") -> CheckMatrices:
    """Check matrix, observable matrix and priors of an (undecomposed) DEM.

    Args:
        dem: Detector error model. ``^`` separators are ignored (components
            are XOR-ed back into the full mechanism).
        mode: ``"mechanism"``, ``"dominant"`` or ``"ldpc"`` (module docstring).
    """
    if mode not in MERGE_MODES:
        raise ValueError(f"mode must be one of {MERGE_MODES}.")
    num_obs = max(dem.num_observables, 1)
    if mode == "ldpc":
        from ldpc.ckt_noise.dem_matrices import (
            detector_error_model_to_check_matrices)
        m = detector_error_model_to_check_matrices(
            dem, allow_undecomposed_hyperedges=True)
        h = csc_matrix(m.check_matrix, dtype=np.uint8)
        lmat = csc_matrix(m.observables_matrix, dtype=np.uint8)
        keep = np.flatnonzero(np.diff(h.indptr) > 0)
        undetectable = 0.0
        for j in np.setdiff1d(np.arange(h.shape[1]), keep):
            if lmat[0, j]:
                undetectable = xor_probability(undetectable, m.priors[j])
        return CheckMatrices(h[:, keep], lmat[:, keep],
                             np.asarray(m.priors)[keep], mode, undetectable)

    undetectable = 0.0
    if mode == "mechanism":
        merged: dict[tuple[frozenset, frozenset], float] = {}
        for prob, components in _dem_errors(dem):
            dets, obs = _xor_sets(components)
            if not dets:
                if 0 in obs:
                    undetectable = xor_probability(undetectable, prob)
                continue
            key = (dets, obs)
            merged[key] = xor_probability(merged.get(key, 0.0), prob)
        columns = [(d, o, p) for (d, o), p in merged.items()]
        return _assemble(columns, dem.num_detectors, num_obs, mode,
                         undetectable)

    # mode == "dominant"
    classes: dict[frozenset, dict[frozenset, float]] = {}
    for prob, components in _dem_errors(dem):
        dets, obs = _xor_sets(components)
        if not dets:
            if 0 in obs:
                undetectable = xor_probability(undetectable, prob)
            continue
        per_class = classes.setdefault(dets, {})
        per_class[obs] = xor_probability(per_class.get(obs, 0.0), prob)
    columns = []
    for dets, per_class in classes.items():
        total = 0.0
        for prob in per_class.values():
            total = xor_probability(total, prob)
        best = max(per_class.items(), key=lambda kv: kv[1])[0]
        columns.append((dets, best, total))
    return _assemble(columns, dem.num_detectors, num_obs, mode, undetectable)


def graph_check_matrices(decomposed_dem: stim.DetectorErrorModel
                         ) -> CheckMatrices:
    """Matching graph of a DEM built with ``decompose_errors=True``.

    Each graphlike component (1 or 2 detectors) of each mechanism becomes an
    edge with the mechanism's probability; parallel edges are XOR-merged and
    keep the observable of their most probable class. Components without
    detectors are dropped (their observable effect is reported as
    ``undetectable_probability`` when they flip observable 0).

    Raises:
        ValueError: if a component has more than two detectors.
    """
    classes: dict[frozenset, dict[frozenset, float]] = {}
    undetectable = 0.0
    for prob, components in _dem_errors(decomposed_dem):
        for dets, obs in components:
            if len(dets) > 2:
                raise ValueError("DEM component with more than two detectors;"
                                 " build the DEM with decompose_errors=True.")
            if not dets:
                if 0 in obs:
                    undetectable = xor_probability(undetectable, prob)
                continue
            per_class = classes.setdefault(dets, {})
            per_class[obs] = xor_probability(per_class.get(obs, 0.0), prob)
    columns = []
    for dets, per_class in classes.items():
        total = 0.0
        for prob in per_class.values():
            total = xor_probability(total, prob)
        best = max(per_class.items(), key=lambda kv: kv[1])[0]
        columns.append((dets, best, total))
    return _assemble(columns, decomposed_dem.num_detectors,
                     max(decomposed_dem.num_observables, 1), "graph",
                     undetectable)


def unique_rows(detection_events: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Distinct rows of a boolean matrix and the inverse index."""
    events = np.asarray(detection_events, dtype=bool)
    if events.shape[1] == 0:
        return events[:1], np.zeros(events.shape[0], dtype=np.int64)
    packed = np.packbits(events, axis=1)
    view = np.ascontiguousarray(packed).view(
        np.dtype((np.void, packed.shape[1])))[:, 0]
    _, first, inverse = np.unique(view, return_index=True,
                                  return_inverse=True)
    return events[first], inverse.reshape(-1)


class LdpcLrbDecoder:
    """One ldpc decoder bound to a ``CheckMatrices`` problem.

    Args:
        matrices: Decoding problem.
        decode_fn: ``syndrome (uint8) -> error vector`` for one shot.
        name: Label used in reports.

    Attributes:
        decode_seconds: Wall time spent in ``decode_fn`` so far.
        decode_calls: Number of ``decode_fn`` calls so far.
    """

    def __init__(self, matrices: CheckMatrices,
                 decode_fn: Callable[[np.ndarray], np.ndarray], name: str):
        self.matrices = matrices
        self._decode_fn = decode_fn
        self.name = name
        self._obs = matrices.observables_matrix.toarray().astype(np.int64)
        self.decode_seconds = 0.0
        self.decode_calls = 0
        self.unsatisfied = 0

    def decode_errors(self, syndrome: np.ndarray) -> np.ndarray:
        """Error vector (over columns) for one syndrome."""
        syndrome = np.ascontiguousarray(syndrome, dtype=np.uint8)
        start = time.perf_counter()
        errors = self._decode_fn(syndrome)
        self.decode_seconds += time.perf_counter() - start
        self.decode_calls += 1
        return np.asarray(errors, dtype=np.uint8)

    def predict(self, detection_events: np.ndarray, dedup: bool = True,
                check_syndrome: bool = True) -> np.ndarray:
        """Predicted flip of observable 0 for each shot.

        Args:
            detection_events: ``(shots, num_detectors)`` booleans.
            dedup: Decode each distinct syndrome once.
            check_syndrome: Count decodings whose error vector does not
                reproduce the syndrome (``self.unsatisfied``).
        """
        events = np.asarray(detection_events, dtype=bool)
        if dedup:
            rows, inverse = unique_rows(events)
        else:
            rows, inverse = events, np.arange(events.shape[0])
        h = self.matrices.check_matrix
        out = np.zeros(rows.shape[0], dtype=bool)
        for i, row in enumerate(rows):
            if not row.any():
                continue
            errors = self.decode_errors(row.astype(np.uint8))
            if check_syndrome:
                if not np.array_equal((h @ errors) % 2, row.astype(np.uint8)):
                    self.unsatisfied += 1
            out[i] = (self._obs[0] @ errors.astype(np.int64)) % 2 == 1
        return out[inverse]


def _bp_kwargs(matrices: CheckMatrices, max_iter: int, bp_method: str,
               schedule: str, ms_scaling_factor: float) -> dict:
    return dict(error_channel=[float(x) for x in matrices.priors],
                max_iter=max_iter, bp_method=bp_method, schedule=schedule,
                ms_scaling_factor=ms_scaling_factor)


def make_bposd(matrices: CheckMatrices, *, max_iter: int = 30,
               bp_method: str = "product_sum", schedule: str = "parallel",
               ms_scaling_factor: float = 1.0, osd_method: str = "OSD_CS",
               osd_order: int = 7, name: Optional[str] = None
               ) -> LdpcLrbDecoder:
    """BP+OSD (``ldpc.BpOsdDecoder``)."""
    from ldpc import BpOsdDecoder
    decoder = BpOsdDecoder(matrices.check_matrix,
                           osd_method=osd_method, osd_order=osd_order,
                           **_bp_kwargs(matrices, max_iter, bp_method,
                                        schedule, ms_scaling_factor))
    label = name or f"bposd_{osd_method.lower()}{osd_order}_{matrices.mode}"
    return LdpcLrbDecoder(matrices, decoder.decode, label)


def make_bplsd(matrices: CheckMatrices, *, max_iter: int = 30,
               bp_method: str = "product_sum", schedule: str = "parallel",
               ms_scaling_factor: float = 1.0, lsd_method: str = "LSD_CS",
               lsd_order: int = 7, bits_per_step: int = 1,
               always_run_lsd: bool = False, name: Optional[str] = None
               ) -> LdpcLrbDecoder:
    """BP+LSD (``ldpc.BpLsdDecoder``)."""
    from ldpc import BpLsdDecoder
    decoder = BpLsdDecoder(matrices.check_matrix, lsd_method=lsd_method,
                           lsd_order=lsd_order, bits_per_step=bits_per_step,
                           always_run_lsd=always_run_lsd,
                           **_bp_kwargs(matrices, max_iter, bp_method,
                                        schedule, ms_scaling_factor))
    label = name or f"bplsd_{lsd_method.lower()}{lsd_order}_{matrices.mode}"
    return LdpcLrbDecoder(matrices, decoder.decode, label)


def make_union_find(matrices: CheckMatrices, *, peeling: Optional[bool] = None,
                    weighted: bool = True, name: Optional[str] = None
                    ) -> LdpcLrbDecoder:
    """Union-find (``ldpc.UnionFindDecoder``).

    Args:
        matrices: Problem; peeling requires every column weight to be 1 or 2
            (e.g. ``graph_check_matrices``).
        peeling: Use ldpc's peeling (graph) mode. Off by default: in ldpc
            2.4.1 it hangs on some valid high-weight syndromes of these
            graphs (use ``DelfosseNickersonUF`` for textbook peeling UF).
        weighted: Grow clusters one column at a time in order of
            ``log((1-p)/p)``; otherwise add all neighbours at once.
    """
    from ldpc import UnionFindDecoder
    weights = np.diff(matrices.check_matrix.indptr)
    graphlike = bool(weights.size) and int(weights.max()) <= 2 and int(
        weights.min()) >= 1
    if peeling is None:
        peeling = False
    if peeling and not graphlike:
        raise ValueError("Peeling union-find needs column weights 1 or 2; "
                         "ldpc would abort the interpreter.")
    # ldpc 2.4.1: the argument is typed str; "" selects peeling, any
    # non-empty string selects matrix-solve ("inversion") mode.
    decoder = UnionFindDecoder(matrices.check_matrix,
                               uf_method="" if peeling else "inversion")
    llrs = matrices.llrs() if weighted else None
    bits_per_step = 1 if weighted else 0

    def decode(syndrome: np.ndarray) -> np.ndarray:
        return decoder.decode(syndrome, llrs=llrs,
                              bits_per_step=bits_per_step)

    label = name or ("uf_" + ("peel" if peeling else "matrix") + "_"
                     + ("w" if weighted else "u") + "_" + matrices.mode)
    return LdpcLrbDecoder(matrices, decode, label)


def make_belief_find(matrices: CheckMatrices, *, max_iter: int = 30,
                     bp_method: str = "product_sum",
                     schedule: str = "parallel", uf_method: str = "inversion",
                     bits_per_step: int = 1, name: Optional[str] = None
                     ) -> LdpcLrbDecoder:
    """BP followed by union-find on failure (``ldpc.BeliefFindDecoder``)."""
    from ldpc import BeliefFindDecoder
    decoder = BeliefFindDecoder(matrices.check_matrix, uf_method=uf_method,
                                bits_per_step=bits_per_step,
                                **_bp_kwargs(matrices, max_iter, bp_method,
                                             schedule, 1.0))
    label = name or f"bf_{uf_method}_{matrices.mode}"
    return LdpcLrbDecoder(matrices, decoder.decode, label)


class DelfosseNickersonUF:
    """Textbook union-find decoder on a matching graph (pure Python).

    Delfosse and Nickerson (arXiv:1709.06218): every cluster with odd
    syndrome parity that does not touch the boundary grows by half an edge
    in all directions per round; fully grown edges fuse clusters; growth
    stops when no odd, boundary-free cluster is left. The fully grown edges
    form an erasure, which is decoded by peeling a spanning forest (rooted at
    the boundary where a cluster touches it). The spanning forest is built by
    Kruskal's algorithm on the most probable edges first; the original paper
    allows any spanning forest.

    Written because ldpc 2.4.1's peeling ``UnionFindDecoder`` hangs on some
    valid high-weight syndromes of these graphs (and aborted a worker with a
    C++ ``Iterator index out of bounds`` exception).

    Args:
        matrices: Graphlike problem (every column of weight 1 or 2), e.g.
            ``graph_check_matrices``.
    """

    def __init__(self, matrices: CheckMatrices):
        h = matrices.check_matrix.tocsc()
        self.num_nodes = h.shape[0]
        self.boundary = self.num_nodes
        self.edges = []
        for j in range(h.shape[1]):
            dets = h.indices[h.indptr[j]:h.indptr[j + 1]]
            if len(dets) == 1:
                self.edges.append((int(dets[0]), self.boundary))
            elif len(dets) == 2:
                self.edges.append((int(dets[0]), int(dets[1])))
            else:
                raise ValueError("Delfosse-Nickerson UF needs a graph.")
        self.adjacency: list[list[int]] = [[] for _ in range(
            self.num_nodes + 1)]
        for j, (u, v) in enumerate(self.edges):
            self.adjacency[u].append(j)
            self.adjacency[v].append(j)
        # Kruskal order: most probable (lowest weight) edges first.
        self.kruskal_order = np.argsort(matrices.llrs(), kind="stable")

    def decode(self, syndrome: np.ndarray) -> np.ndarray:
        n_edges = len(self.edges)
        support = np.zeros(n_edges, dtype=np.int8)
        parent: dict[int, int] = {}
        members: dict[int, list[int]] = {}
        parity: dict[int, int] = {}

        def find(x: int) -> int:
            root = x
            while parent[root] != root:
                root = parent[root]
            while parent[x] != root:
                parent[x], x = root, parent[x]
            return root

        def add(x: int, odd: int) -> None:
            if x not in parent:
                parent[x] = x
                members[x] = [x]
                parity[x] = odd

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            if len(members[ra]) < len(members[rb]):
                ra, rb = rb, ra
            parent[rb] = ra
            members[ra].extend(members.pop(rb))
            parity[ra] ^= parity.pop(rb)

        defects = [int(v) for v in np.flatnonzero(syndrome)]
        for v in defects:
            add(v, 1)
        add(self.boundary, 0)

        def odd_roots() -> list[int]:
            broot = find(self.boundary)
            return [r for r in members if parity[r] and r != broot]

        active = odd_roots()
        while active:
            grown: list[int] = []
            progress = False
            for root in active:
                frontier = set()
                for v in members[root]:
                    if v == self.boundary:
                        continue
                    for j in self.adjacency[v]:
                        if support[j] < 2:
                            frontier.add(j)
                for j in frontier:
                    support[j] += 1
                    progress = True
                    if support[j] == 2:
                        grown.append(j)
            if not progress:  # odd component without boundary: invalid input
                break
            for j in grown:
                u, v = self.edges[j]
                add(u, 0)
                add(v, 0)
                union(u, v)
            active = odd_roots()

        # Peeling decoder on the erasure (fully grown edges).
        erasure = [j for j in self.kruskal_order if support[j] == 2]
        forest_parent: dict[int, int] = {}
        forest = {}
        tree = {}

        def tfind(x: int) -> int:
            while tree.setdefault(x, x) != x:
                tree[x] = tree[tree[x]]
                x = tree[x]
            return x

        neighbours: dict[int, list[tuple[int, int]]] = {}
        for j in erasure:
            u, v = self.edges[j]
            ru, rv = tfind(u), tfind(v)
            if ru == rv:
                continue
            tree[ru] = rv
            neighbours.setdefault(u, []).append((v, j))
            neighbours.setdefault(v, []).append((u, j))
        # Root each tree at the boundary when present, else at any vertex.
        order: list[int] = []
        seen: set[int] = set()
        roots = ([self.boundary] if self.boundary in neighbours else []) + [
            v for v in neighbours if v != self.boundary]
        for root in roots:
            if root in seen:
                continue
            seen.add(root)
            stack = [root]
            while stack:
                x = stack.pop()
                order.append(x)
                for y, j in neighbours.get(x, []):
                    if y not in seen:
                        seen.add(y)
                        forest_parent[y] = x
                        forest[y] = j
                        stack.append(y)
        marks = np.zeros(self.num_nodes + 1, dtype=np.uint8)
        marks[:self.num_nodes] = syndrome
        correction = np.zeros(n_edges, dtype=np.uint8)
        for x in reversed(order):
            if x in forest_parent and marks[x]:
                correction[forest[x]] ^= 1
                marks[x] ^= 1
                marks[forest_parent[x]] ^= 1
        return correction


def make_dn_union_find(matrices: CheckMatrices, name: str = "uf_dn"
                       ) -> LdpcLrbDecoder:
    """``DelfosseNickersonUF`` wrapped with the common ``predict`` interface."""
    decoder = DelfosseNickersonUF(matrices)
    return LdpcLrbDecoder(matrices, decoder.decode, name)


class MatchingControl:
    """Minimum-weight matching (PyMatching) on a ``graph_check_matrices`` graph.

    Not one of the work-package decoders: a control that separates what the
    decomposed graph loses from what the union-find heuristic loses. It is
    vectorized (``decode_batch``) and has the ``predict`` interface of
    ``LdpcLrbDecoder``.
    """

    def __init__(self, matrices: CheckMatrices, name: str = "mwpm_graph"):
        import pymatching
        self.matrices = matrices
        self.name = name
        self._matching = pymatching.Matching.from_check_matrix(
            matrices.check_matrix, weights=matrices.llrs(),
            faults_matrix=matrices.observables_matrix)
        self.decode_seconds = 0.0
        self.decode_calls = 0
        self.unsatisfied = 0

    def predict(self, detection_events: np.ndarray, dedup: bool = False
                ) -> np.ndarray:
        events = np.asarray(detection_events, dtype=np.uint8)
        start = time.perf_counter()
        out = self._matching.decode_batch(events)[:, 0].astype(bool)
        self.decode_seconds += time.perf_counter() - start
        self.decode_calls += events.shape[0]
        return out


# Standard decoder set used by scripts/qubit512/ldpc_lut_vs_mld.py. Settings
# were chosen on held-out Clifford sequences 10 and 11 (evaluation uses 0-9):
# product-sum BP beat min-sum, 5 BP iterations matched 30 for OSD and beat
# 30 for LSD and belief-find, and OSD-CS order 15 matched order 7. At depth
# 14 the parallel schedule often converged to improbable multi-mechanism
# solutions (OSD/LSD then never run); the serial schedule and
# always_run_lsd variants are included for that reason.
STANDARD_DECODERS = (
    "bposd_cs7", "bposd_cs7_serial", "bposd_0", "bplsd_cs7",
    "bplsd_cs7_always", "bposd_cs7_dominant", "bposd_cs7_ldpcmatrices",
    "uf_dn", "uf_graph", "uf_hyper", "belief_find")
BP_ITERATIONS = 5


def build_standard_decoders(circuit: stim.Circuit,
                            names=STANDARD_DECODERS,
                            dem: Optional[stim.DetectorErrorModel] = None
                            ) -> dict[str, LdpcLrbDecoder]:
    """Construct the named decoders for one noisy circuit.

    Names (BP is product-sum with ``BP_ITERATIONS`` parallel iterations):
        ``bposd_cs7``: BP (parallel schedule) + OSD-CS order 7 on
            ``"mechanism"`` columns. OSD runs only when BP fails to converge.
        ``bposd_cs7_serial``: same with the serial BP schedule.
        ``bposd_0``: parallel BP + OSD-0.
        ``bplsd_cs7``: BP + LSD-CS order 7 on ``"mechanism"`` columns (LSD
            only when BP fails).
        ``bplsd_cs7_always``: same with ``always_run_lsd=True``.
        ``bposd_cs7_dominant``: BP+OSD-CS7 on ``"dominant"`` columns.
        ``bposd_cs7_ldpcmatrices``: BP+OSD-CS7 on ldpc's own DEM conversion
            (the ``SinterBpOsdDecoder`` path; last-listed observable wins).
        ``uf_dn``: ``DelfosseNickersonUF`` on the decomposed graph.
        ``uf_graph``: ldpc matrix-solve union-find on the decomposed graph,
            weighted growth (ldpc's peeling mode hangs on some syndromes).
        ``uf_hyper``: ldpc weighted matrix-solve union-find on
            ``"mechanism"`` columns (hyperedges kept).
        ``belief_find``: BP + matrix-solve union-find on ``"mechanism"``.
    """
    dem = circuit.detector_error_model() if dem is None else dem
    cache: dict[str, CheckMatrices] = {}

    def matrices(mode: str) -> CheckMatrices:
        if mode not in cache:
            if mode == "graph":
                cache[mode] = graph_check_matrices(
                    circuit.detector_error_model(decompose_errors=True))
            else:
                cache[mode] = dem_check_matrices(dem, mode)
        return cache[mode]

    it = BP_ITERATIONS
    builders = {
        "bposd_cs7": lambda: make_bposd(matrices("mechanism"), max_iter=it,
                                        name="bposd_cs7"),
        "bposd_cs7_serial": lambda: make_bposd(
            matrices("mechanism"), max_iter=it, schedule="serial",
            name="bposd_cs7_serial"),
        "bplsd_cs7_always": lambda: make_bplsd(
            matrices("mechanism"), max_iter=it, always_run_lsd=True,
            name="bplsd_cs7_always"),
        "bposd_0": lambda: make_bposd(matrices("mechanism"), max_iter=it,
                                      osd_method="OSD_0", osd_order=0,
                                      name="bposd_0"),
        "bplsd_cs7": lambda: make_bplsd(matrices("mechanism"), max_iter=it,
                                        name="bplsd_cs7"),
        "bposd_cs7_dominant": lambda: make_bposd(
            matrices("dominant"), max_iter=it, name="bposd_cs7_dominant"),
        "bposd_cs7_ldpcmatrices": lambda: make_bposd(
            matrices("ldpc"), max_iter=it, name="bposd_cs7_ldpcmatrices"),
        "uf_dn": lambda: make_dn_union_find(matrices("graph"),
                                            name="uf_dn"),
        "uf_graph": lambda: make_union_find(matrices("graph"), peeling=False,
                                            weighted=True, name="uf_graph"),
        "uf_hyper": lambda: make_union_find(matrices("mechanism"),
                                            peeling=False, weighted=True,
                                            name="uf_hyper"),
        "belief_find": lambda: make_belief_find(matrices("mechanism"),
                                                max_iter=it,
                                                name="belief_find"),
    }
    unknown = [n for n in names if n not in builders]
    if unknown:
        raise ValueError(f"Unknown decoders {unknown}.")
    return {n: builders[n]() for n in names}
