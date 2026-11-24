"""Matching (MWPM) decoders for the qubit ``[[5,1,2]]`` LRB circuits.

Work package B of the qubit-512 correction-decoders study. Every decoder reads
the detection events of the ``data5`` circuit built by
``circuits.build_lrb_circuit`` and predicts the flip of observable 0 (X_L).

PyMatching decoders (PyMatching 2.3.1, Stim 1.15)
    ``PyMatchingDecoder``
        Plain MWPM on Stim's decomposed DEM
        (``detector_error_model(decompose_errors=True)``). With
        ``correlated=True`` it is PyMatching's two-pass correlated matching
        (``enable_correlations=True``, Fowler arXiv:1310.0863 per the
        PyMatching docstring). PyMatching refuses decomposed instructions that
        contain a component with no detectors, and Stim emits such components
        for these circuits (see ``strip_undetectable_components``). The
        correlated decoder is therefore built from the stripped DEM, which
        gives exactly the graph that plain PyMatching builds from the
        unstripped DEM.
    ``ObservableDetectorMatching``
        Complementary gap: observable 0 becomes an extra detector ``D_L`` and
        Stim decomposes the resulting DEM. Each shot is decoded twice, with
        ``D_L`` forced to 0 and to 1. ``W(l)`` is the weight of the minimum
        matching with ``D_L = l``. The decision is ``argmin_l W(l)`` and the
        gap is ``|W(1) - W(0)|``. On its own graph this is exact, because
        every component that touches ``D_L`` is graphlike. Optionally it uses
        correlated matching in both passes; that variant is heuristic,
        because the two passes are reweighted differently.

Exact min-sum decoders
    ``MinSumTransferMatrix``
        The (min, +) version of ``mld.TransferMatrixMLD``. For any list of
        mechanisms ``(detectors, flips_L, weight)`` that each touch at most
        two consecutive time slots, it returns ``W(l)``: the exact minimum
        total weight of a mechanism subset that reproduces the observed
        detection events and has logical flip ``l``. It is used three ways.
        - On PyMatching's own graph (``mechanisms_from_matching``) it gives
          the exact complementary gap of plain MWPM. Its ``argmin`` is the
          MWPM decision up to weight ties.
        - On the class-aware graph (``classaware_mechanisms``), parallel
          components with different observables are kept as separate edges.
          This removes PyMatching's "first edge's observable wins" rule.
        - On the undecomposed DEM (``mechanisms_from_dem``) it is the exact
          minimum-weight-error (MWE) decoder on the hypergraph, with no
          decomposition at all.

Weights are log-likelihood ratios ``ln((1 - q) / q)`` throughout, so all
gaps are in nats, the same unit as the exact MLD ``|LLR|``.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import pymatching
import stim

from . import mld as M

# Two class weights closer than this are a tie (PyMatching discretizes its
# weights, so its returned weights are only accurate to about 1e-6).
TIE_TOL = 1e-9
PYMATCHING_WEIGHT_TOL = 1e-4

Mechanism = tuple[tuple[int, ...], bool, float]


# ---------------------------------------------------------------------------
# DEM utilities
# ---------------------------------------------------------------------------

def error_components(instruction: stim.DemInstruction
                     ) -> list[tuple[tuple[int, ...], int]]:
    """Split an ``error`` instruction at its ``^`` separators.

    Returns one ``(sorted detector ids, observable bitmask)`` per component.
    """
    comps: list[tuple[list[int], int]] = [([], 0)]
    for target in instruction.targets_copy():
        if target.is_separator():
            comps.append(([], 0))
        elif target.is_relative_detector_id():
            comps[-1][0].append(target.val)
        elif target.is_logical_observable_id():
            dets, mask = comps[-1]
            comps[-1] = (dets, mask ^ (1 << target.val))
    return [(tuple(sorted(d)), m) for d, m in comps]


def _symmetric_difference(components: Iterable[tuple[tuple[int, ...], int]]
                          ) -> tuple[tuple[int, ...], int]:
    dets: set[int] = set()
    mask = 0
    for d, m in components:
        dets ^= set(d)
        mask ^= m
    return tuple(sorted(dets)), mask


def strip_undetectable_components(dem: stim.DetectorErrorModel
                                  ) -> tuple[stim.DetectorErrorModel, int]:
    """Drop the components with no detector from decomposed instructions.

    Stim's decomposer sometimes writes a hyperedge as known graphlike pieces
    plus an observable-only remainder, e.g. ``D6 D8 L0 ^ D10 ^ L0`` for a
    mechanism that flips ``D6 D8 D10`` and not ``L0``. Plain PyMatching
    silently ignores the remainder, so its graph attributes ``L0`` to the
    ``D6 D8`` piece. Correlated PyMatching raises ``ValueError`` instead.
    This function makes the DEM that plain PyMatching effectively uses.

    Returns:
        The stripped (flattened) DEM and the number of instructions changed.
    """
    out = stim.DetectorErrorModel()
    changed = 0
    for instruction in dem.flattened():
        if instruction.type != "error":
            out.append(instruction)
            continue
        groups: list[list[stim.DemTarget]] = [[]]
        for target in instruction.targets_copy():
            if target.is_separator():
                groups.append([])
            else:
                groups[-1].append(target)
        keep = [g for g in groups
                if any(t.is_relative_detector_id() for t in g)]
        if len(keep) == len(groups):
            out.append(instruction)
            continue
        changed += 1
        targets: list[stim.DemTarget] = []
        for index, group in enumerate(keep):
            if index:
                targets.append(stim.target_separator())
            targets.extend(group)
        if targets:
            out.append("error", instruction.args_copy(), targets)
    return out, changed


def observable_as_detector_circuit(circuit: stim.Circuit, observable: int = 0
                                   ) -> tuple[stim.Circuit, int]:
    """Replace ``OBSERVABLE_INCLUDE(observable)`` by a ``DETECTOR``.

    The detector reads the same measurement records, so its noiseless value
    is 0 and it fires exactly when the observable flips.

    Returns:
        The new circuit (no observables left if ``observable`` was the only
        one) and the index of the new detector.
    """
    out = stim.Circuit()
    detectors_seen = 0
    new_index: Optional[int] = None
    for instruction in circuit.flattened():
        if (instruction.name == "OBSERVABLE_INCLUDE"
                and int(instruction.gate_args_copy()[0]) == observable):
            if new_index is not None:
                raise ValueError("Observable included more than once.")
            new_index = detectors_seen
            out.append("DETECTOR", instruction.targets_copy(), [-1, -1])
            detectors_seen += 1
            continue
        if instruction.name == "DETECTOR":
            detectors_seen += 1
        out.append(instruction)
    if new_index is None:
        raise ValueError(f"Observable {observable} not found.")
    return out, new_index


def _weight(probability: float) -> float:
    return math.log((1.0 - probability) / probability)


def _xor_probability(p: float, q: float) -> float:
    return p * (1.0 - q) + q * (1.0 - p)


def mechanisms_from_dem(dem: stim.DetectorErrorModel) -> list[Mechanism]:
    """Undecomposed DEM lines as min-sum mechanisms (exact hypergraph)."""
    out = []
    for mech in M.dem_mechanisms(dem):
        if mech.probability <= 0.0:
            continue
        if not mech.detectors and not mech.flips_observable:
            continue
        out.append((mech.detectors, mech.flips_observable,
                    _weight(mech.probability)))
    return out


def mechanisms_from_matching(matching: pymatching.Matching,
                             observable: int = 0) -> list[Mechanism]:
    """Edges of a PyMatching graph (boundary edges have one detector)."""
    out = []
    for u, v, data in matching.edges():
        dets = (int(u),) if v is None else tuple(sorted((int(u), int(v))))
        out.append((dets, observable in data["fault_ids"],
                    float(data["weight"])))
    return out


def classaware_mechanisms(dem_decomposed: stim.DetectorErrorModel
                          ) -> list[Mechanism]:
    """Graphlike components merged per (detectors, observable) class.

    Like PyMatching's graph, but parallel components that disagree on the
    observable stay separate edges, so neither logical class is lost.
    Components without detectors are dropped, as plain PyMatching does.
    """
    merged: "OrderedDict[tuple[tuple[int, ...], bool], float]" = OrderedDict()
    for instruction in dem_decomposed.flattened():
        if instruction.type != "error":
            continue
        probability = instruction.args_copy()[0]
        for dets, mask in error_components(instruction):
            if not dets:
                continue
            if len(dets) > 2:
                raise ValueError("Component is not graphlike.")
            key = (dets, bool(mask & 1))
            merged[key] = _xor_probability(merged.get(key, 0.0), probability)
    return [(d, flips, _weight(q)) for (d, flips), q in merged.items()]


def obsdet_mechanisms(matching: pymatching.Matching, obs_detector: int
                      ) -> list[Mechanism]:
    """Edges of an observable-as-detector graph, mapped back to (dets, L)."""
    out = []
    for u, v, data in matching.edges():
        nodes = [int(u)] + ([] if v is None else [int(v)])
        flips = obs_detector in nodes
        dets = tuple(sorted(n for n in nodes if n != obs_detector))
        out.append((dets, flips, float(data["weight"])))
    return out


# ---------------------------------------------------------------------------
# Census of the decomposition
# ---------------------------------------------------------------------------

@dataclass
class EdgeContributions:
    """Contributions of decomposed components to one PyMatching edge."""

    kept_flips: bool
    probabilities: list[float] = field(default_factory=list)
    flips: list[bool] = field(default_factory=list)

    @property
    def merged_probability(self) -> float:
        q = 0.0
        for p in self.probabilities:
            q = _xor_probability(q, p)
        return q

    @property
    def conflict(self) -> bool:
        return any(f != self.kept_flips for f in self.flips)

    @property
    def lost_probability(self) -> float:
        """Probability mass of contributions whose observable was dropped."""
        return sum(p for p, f in zip(self.probabilities, self.flips)
                   if f != self.kept_flips)


def pymatching_merge_model(dem_decomposed: stim.DetectorErrorModel
                           ) -> "OrderedDict[tuple[int, ...], EdgeContributions]":
    """Reproduce how plain PyMatching merges decomposed components.

    Components are added in DEM order; zero-detector components are ignored;
    parallel edges are merged assuming independence and keep the observable
    of the first added component (PyMatching 2.3.1 docstring of
    ``Matching.from_detector_error_model``).
    """
    edges: "OrderedDict[tuple[int, ...], EdgeContributions]" = OrderedDict()
    for instruction in dem_decomposed.flattened():
        if instruction.type != "error":
            continue
        probability = instruction.args_copy()[0]
        for dets, mask in error_components(instruction):
            if not dets:
                continue
            flips = bool(mask & 1)
            if dets not in edges:
                edges[dets] = EdgeContributions(kept_flips=flips)
            edges[dets].probabilities.append(probability)
            edges[dets].flips.append(flips)
    return edges


@dataclass
class DecompositionCensus:
    """How Stim decomposes one circuit's DEM and what PyMatching keeps."""

    num_detectors: int
    mechanisms: int
    mass: float
    hyperedges: int
    hyperedge_mass: float
    max_hyperedge_size: int
    decomposed_lines: int
    graphlike_kept_lines: int
    graphlike_split_lines: int
    graphlike_split_mass: float
    hyper_clean_lines: int
    hyper_clean_mass: float
    undetectable_component_lines: int
    undetectable_component_mass: float
    bulk_observable_components: int
    pm_edges: int
    pm_boundary_edges: int
    pm_merged_edges: int
    pm_conflict_edges: int
    pm_conflict_lost_mass: float
    pm_conflict_kept_minority: int
    obsdet_lines: int
    obsdet_hyper_lines: int
    obsdet_l_components: int
    obsdet_l_only_components: int

    def as_row(self) -> dict:
        return dict(self.__dict__)


def decomposition_census(circuit: stim.Circuit) -> DecompositionCensus:
    """Count non-graphlike mechanisms and how Stim/PyMatching handle them."""
    und = circuit.detector_error_model()
    dec = circuit.detector_error_model(decompose_errors=True)
    hyper = [m for m in M.dem_mechanisms(und) if len(m.detectors) > 2]
    lines = [ins for ins in dec.flattened() if ins.type == "error"]
    kept = split = hyper_clean = undetectable = bulk_l = 0
    split_mass = hyper_clean_mass = undetectable_mass = 0.0
    for instruction in lines:
        probability = instruction.args_copy()[0]
        comps = error_components(instruction)
        total, _ = _symmetric_difference(comps)
        has_empty = any(not d for d, _ in comps)
        bulk_l += sum(1 for d, m in comps if len(d) == 2 and m & 1)
        if has_empty:
            undetectable += 1
            undetectable_mass += probability
        elif len(total) <= 2 and len(comps) == 1:
            kept += 1
        elif len(total) <= 2:
            split += 1
            split_mass += probability
        else:
            hyper_clean += 1
            hyper_clean_mass += probability
    merge = pymatching_merge_model(dec)
    conflicts = [e for e in merge.values() if e.conflict]
    minority = 0
    for e in conflicts:
        kept_mass = sum(p for p, f in zip(e.probabilities, e.flips)
                        if f == e.kept_flips)
        minority += int(kept_mass < e.lost_probability)
    obs_circuit, obs_det = observable_as_detector_circuit(circuit)
    obs_dem = obs_circuit.detector_error_model(decompose_errors=True)
    obs_lines = obs_hyper = obs_l = obs_l_only = 0
    for instruction in obs_dem.flattened():
        if instruction.type != "error":
            continue
        obs_lines += 1
        comps = error_components(instruction)
        total, _ = _symmetric_difference(comps)
        obs_hyper += int(len(total) > 2)
        for dets, _ in comps:
            if obs_det in dets:
                obs_l += 1
                obs_l_only += int(len(dets) == 1)
    return DecompositionCensus(
        num_detectors=und.num_detectors,
        mechanisms=sum(1 for _ in M.dem_mechanisms(und)),
        mass=sum(m.probability for m in M.dem_mechanisms(und)),
        hyperedges=len(hyper),
        hyperedge_mass=sum(m.probability for m in hyper),
        max_hyperedge_size=max((len(m.detectors) for m in hyper), default=0),
        decomposed_lines=len(lines),
        graphlike_kept_lines=kept,
        graphlike_split_lines=split,
        graphlike_split_mass=split_mass,
        hyper_clean_lines=hyper_clean,
        hyper_clean_mass=hyper_clean_mass,
        undetectable_component_lines=undetectable,
        undetectable_component_mass=undetectable_mass,
        bulk_observable_components=bulk_l,
        pm_edges=len(merge),
        pm_boundary_edges=sum(1 for d in merge if len(d) == 1),
        pm_merged_edges=sum(1 for e in merge.values()
                            if len(e.probabilities) > 1),
        pm_conflict_edges=len(conflicts),
        pm_conflict_lost_mass=sum(e.lost_probability for e in conflicts),
        pm_conflict_kept_minority=minority,
        obsdet_lines=obs_lines,
        obsdet_hyper_lines=obs_hyper,
        obsdet_l_components=obs_l,
        obsdet_l_only_components=obs_l_only,
    )


# ---------------------------------------------------------------------------
# PyMatching decoders
# ---------------------------------------------------------------------------

class PyMatchingDecoder:
    """Plain or correlated MWPM with PyMatching on Stim's decomposed DEM.

    Args:
        dem_decomposed: ``circuit.detector_error_model(decompose_errors=True)``.
        correlated: Use two-pass correlated matching.
    """

    def __init__(self, dem_decomposed: stim.DetectorErrorModel,
                 correlated: bool = False):
        self.correlated = correlated
        self.stripped_instructions = 0
        dem = dem_decomposed
        if correlated:
            dem, self.stripped_instructions = strip_undetectable_components(
                dem_decomposed)
        self.matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=correlated)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        """Predicted flip of observable 0 for each shot."""
        events = np.asarray(detection_events, dtype=np.uint8)
        predictions = self.matching.decode_batch(
            events, enable_correlations=self.correlated)
        return predictions[:, 0].astype(bool)

    def predict_with_weights(self, detection_events: np.ndarray
                             ) -> tuple[np.ndarray, np.ndarray]:
        events = np.asarray(detection_events, dtype=np.uint8)
        predictions, weights = self.matching.decode_batch(
            events, return_weights=True,
            enable_correlations=self.correlated)
        return predictions[:, 0].astype(bool), weights


class ObservableDetectorMatching:
    """Complementary gap from matching with observable 0 as a detector.

    Args:
        circuit: Circuit with observable 0 (e.g. ``LRBCircuit.circuit``).
        correlated: Use correlated matching in both forced passes.
    """

    def __init__(self, circuit: stim.Circuit, correlated: bool = False):
        self.correlated = correlated
        obs_circuit, self.obs_detector = observable_as_detector_circuit(
            circuit)
        self.num_detectors = self.obs_detector
        self.dem = obs_circuit.detector_error_model(decompose_errors=True)
        if self.dem.num_observables:
            raise ValueError("Expected no observables after conversion.")
        self.matching = pymatching.Matching.from_detector_error_model(
            self.dem, enable_correlations=correlated)

    def class_weights(self, detection_events: np.ndarray) -> np.ndarray:
        """``(shots, 2)`` matching weights with ``D_L`` forced to 0 and 1."""
        events = np.asarray(detection_events, dtype=np.uint8)
        shots = events.shape[0]
        out = np.empty((shots, 2))
        for value in (0, 1):
            syndrome = np.concatenate(
                [events, np.full((shots, 1), value, dtype=np.uint8)], axis=1)
            _, weights = self.matching.decode_batch(
                syndrome, return_weights=True,
                enable_correlations=self.correlated)
            out[:, value] = weights
        return out

    def decode(self, detection_events: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        """``(predicted_flip, gap)``; ties predict no flip."""
        weights = self.class_weights(detection_events)
        return decision_from_class_weights(weights)


def decision_from_class_weights(weights: np.ndarray, tol: float = TIE_TOL
                                ) -> tuple[np.ndarray, np.ndarray]:
    """Decision ``W(1) < W(0)`` (ties predict no flip) and gap ``|W1-W0|``."""
    diff = weights[:, 1] - weights[:, 0]
    return diff < -tol, np.abs(diff)


# ---------------------------------------------------------------------------
# Exact min-sum transfer matrix
# ---------------------------------------------------------------------------

def _minplus_product(phi: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """``out[i, b] = min_c phi[i, c] + matrix[c, b]`` (row loop; low memory)."""
    out = phi[:, 0:1] + matrix[0][None, :]
    for c in range(1, matrix.shape[0]):
        np.minimum(out, phi[:, c:c + 1] + matrix[c][None, :], out=out)
    return out


def _minplus_update(kernel: np.ndarray, vector: int, weight: float) -> None:
    """In place: ``K(x) <- min(K(x), K(x ^ v) + w)`` (use the edge 0 or 1 times)."""
    if vector == 0:
        if weight < 0:
            kernel += weight
        return
    index = np.arange(kernel.size) ^ vector
    np.minimum(kernel, kernel[index] + weight, out=kernel)


class MinSumTransferMatrix:
    """Exact minimum-weight explanation in each logical class.

    For mechanisms ``(detectors, flips_L, weight)`` that each touch at most
    two consecutive slots, ``class_weights`` returns
    ``W(l) = min { sum_{e in A} w_e : A reproduces the detection events,
    flips_L(A) = l }`` per shot, computed by the same slot sweep as
    ``mld.TransferMatrixMLD`` with (+, x) replaced by (min, +). Each mechanism
    is used at most once. With PyMatching's edge list, ``min_l W(l)`` is the
    MWPM solution weight and ``W(1 - l*) - W(l*)`` the exact complementary
    gap of that graph.

    Args:
        detector_slots: Slot index of every detector (e.g.
            ``mld.lrb_detector_slots``).
        mechanisms: Iterable of ``(detectors, flips_L, weight)``.
    """

    def __init__(self, detector_slots: Sequence[int],
                 mechanisms: Iterable[Mechanism]):
        slots = np.asarray(detector_slots, dtype=np.int64)
        self.num_detectors = len(slots)
        self.num_slots = int(slots.max()) + 1 if len(slots) else 1
        self.slot_detectors = [np.flatnonzero(slots == t)
                               for t in range(self.num_slots)]
        self.slot_sizes = [len(d) for d in self.slot_detectors]
        if max(self.slot_sizes) > 8:
            raise ValueError("Slots larger than 8 detectors are not supported.")
        position = np.zeros(len(slots), dtype=np.int64)
        for dets in self.slot_detectors:
            position[dets] = np.arange(len(dets))
        kernels = []
        for t in range(self.num_slots):
            n_prev = self.slot_sizes[t - 1] if t > 0 else 0
            kernel = np.full(2 ** (n_prev + self.slot_sizes[t] + 1), np.inf)
            kernel[0] = 0.0
            kernels.append(kernel)
        self.num_mechanisms = 0
        for detectors, flips, weight in mechanisms:
            detectors = tuple(int(d) for d in detectors)
            if not detectors:
                if flips:
                    _minplus_update(kernels[0], 1 << self.slot_sizes[0],
                                    weight)
                    self.num_mechanisms += 1
                continue
            mech_slots = slots[list(detectors)]
            t = int(mech_slots.max())
            if int(mech_slots.min()) < t - 1:
                raise ValueError("Mechanism spans non-adjacent slots.")
            n_prev = self.slot_sizes[t - 1] if t > 0 else 0
            n_cur = self.slot_sizes[t]
            vector = 0
            for det in detectors:
                bit = int(position[det])
                vector ^= (1 << (n_prev + bit)) if slots[det] == t else (1 << bit)
            if flips:
                vector ^= 1 << (n_prev + n_cur)
            _minplus_update(kernels[t], vector, weight)
            self.num_mechanisms += 1
        self.initial = kernels[0].copy()
        self.transfer = [None]
        for t in range(1, self.num_slots):
            n_prev = self.slot_sizes[t - 1]
            n_cur = self.slot_sizes[t]
            kernel = kernels[t].reshape(2, 2 ** n_cur, 2 ** n_prev)
            matrix = np.full((2 * 2 ** n_prev, 2 * 2 ** n_cur), np.inf)
            for l_in in range(2):
                for l_out in range(2):
                    matrix[l_in * 2 ** n_prev:(l_in + 1) * 2 ** n_prev,
                           l_out * 2 ** n_cur:(l_out + 1) * 2 ** n_cur] = (
                               kernel[l_in ^ l_out].T)
            self.transfer.append(matrix)

    def _slot_values(self, events: np.ndarray, t: int) -> np.ndarray:
        dets = self.slot_detectors[t]
        weights = (1 << np.arange(len(dets))).astype(np.int64)
        return events[:, dets].astype(np.int64) @ weights

    def _sweep(self, events: np.ndarray) -> np.ndarray:
        shots = events.shape[0]
        psi = np.broadcast_to(self.initial, (shots, self.initial.size)).copy()
        offset = np.zeros(shots)
        for t in range(1, self.num_slots):
            n_prev = self.slot_sizes[t - 1]
            s = self._slot_values(events, t - 1)
            size = 2 ** n_prev
            base = np.arange(2 * size)
            cols = (base[None, :] & (size - 1)) ^ s[:, None]
            cols |= (base[None, :] >> n_prev) << n_prev
            phi = np.take_along_axis(psi, cols, axis=1)
            psi = _minplus_product(phi, self.transfer[t])
            shift = psi.min(axis=1)
            shift = np.where(np.isfinite(shift), shift, 0.0)
            psi -= shift[:, None]
            offset += shift
        n_last = self.slot_sizes[-1]
        s = self._slot_values(events, self.num_slots - 1)
        rows = np.arange(shots)
        return np.stack([psi[rows, s], psi[rows, s + 2 ** n_last]],
                        axis=1) + offset[:, None]

    def class_weights(self, detection_events: np.ndarray,
                      batch_size: int = 16384, unique: bool = True
                      ) -> np.ndarray:
        """``(shots, 2)`` exact minimum weights for ``L = 0`` and ``L = 1``.

        Infinite where no mechanism subset explains the events.
        """
        events = np.asarray(detection_events, dtype=bool)
        if events.shape[1] != self.num_detectors:
            raise ValueError("Wrong number of detector columns.")
        inverse = None
        if unique and events.shape[0] > 1:
            packed = np.packbits(events, axis=1)
            packed, inverse = np.unique(packed, axis=0, return_inverse=True)
            events = np.unpackbits(packed, axis=1,
                                   count=self.num_detectors).astype(bool)
            inverse = inverse.reshape(-1)
        out = np.empty((events.shape[0], 2))
        for start in range(0, events.shape[0], batch_size):
            chunk = events[start:start + batch_size]
            out[start:start + len(chunk)] = self._sweep(chunk)
        return out if inverse is None else out[inverse]

    def decode(self, detection_events: np.ndarray
               ) -> tuple[np.ndarray, np.ndarray]:
        """``(predicted_flip, gap)``; ties predict no flip."""
        return decision_from_class_weights(self.class_weights(detection_events))


def brute_force_class_weights(num_detectors: int,
                              mechanisms: Iterable[Mechanism]) -> np.ndarray:
    """Dense ``W(syndrome, l)`` by min-sum over all syndromes (small cases).

    Index ``i`` encodes detector ``k`` in bit ``k`` and ``l`` in the top bit.
    Independent of any slot structure; used to test ``MinSumTransferMatrix``.
    """
    if num_detectors > 20:
        raise ValueError("Too many detectors for brute force.")
    table = np.full(2 ** (num_detectors + 1), np.inf)
    table[0] = 0.0
    for detectors, flips, weight in mechanisms:
        vector = 0
        for det in detectors:
            vector ^= 1 << int(det)
        if flips:
            vector ^= 1 << num_detectors
        _minplus_update(table, vector, weight)
    return table


# ---------------------------------------------------------------------------
# All decoders for one circuit
# ---------------------------------------------------------------------------

SOFT_DECODERS = ("mld", "mwe", "mwpm_gap", "mwpm_classaware", "mwpm_obs",
                 "mwpm_obs_corr")
HARD_DECODERS = ("none", "mwpm", "mwpm_corr")


@dataclass
class DecoderSuite:
    """Every decoder of this work package for one ``LRBCircuit``.

    Attributes are built lazily by ``build``; ``run`` decodes one batch of
    detection events with all of them and returns predictions and gaps.
    """

    lrb_circuit: object
    mld: M.TransferMatrixMLD = None
    mwe: MinSumTransferMatrix = None
    mwpm: PyMatchingDecoder = None
    mwpm_corr: PyMatchingDecoder = None
    mwpm_gap: MinSumTransferMatrix = None
    mwpm_classaware: MinSumTransferMatrix = None
    mwpm_obs: ObservableDetectorMatching = None
    mwpm_obs_corr: ObservableDetectorMatching = None

    @classmethod
    def build(cls, lrb_circuit) -> "DecoderSuite":
        circuit = lrb_circuit.circuit
        slots = M.lrb_detector_slots(lrb_circuit)
        und = circuit.detector_error_model()
        dec = circuit.detector_error_model(decompose_errors=True)
        suite = cls(lrb_circuit)
        suite.mld = M.TransferMatrixMLD(und, slots)
        suite.mwe = MinSumTransferMatrix(slots, mechanisms_from_dem(und))
        suite.mwpm = PyMatchingDecoder(dec)
        suite.mwpm_corr = PyMatchingDecoder(dec, correlated=True)
        suite.mwpm_gap = MinSumTransferMatrix(
            slots, mechanisms_from_matching(suite.mwpm.matching))
        suite.mwpm_classaware = MinSumTransferMatrix(
            slots, classaware_mechanisms(dec))
        suite.mwpm_obs = ObservableDetectorMatching(circuit)
        suite.mwpm_obs_corr = ObservableDetectorMatching(circuit,
                                                         correlated=True)
        return suite

    def run(self, detection_events: np.ndarray) -> tuple[dict, dict, dict]:
        """Decode a batch with every decoder.

        Returns:
            ``(predictions, gaps, extras)``: predictions per decoder (bool
            arrays), gaps per soft decoder (float arrays, nats) and extras
            (``mld_llr``, ``mwpm_weight`` and the class weights).
        """
        events = np.asarray(detection_events, dtype=bool)
        shots = events.shape[0]
        pred: dict[str, np.ndarray] = {"none": np.zeros(shots, dtype=bool)}
        gaps: dict[str, np.ndarray] = {}
        extras: dict[str, np.ndarray] = {}
        llr = self.mld.log_likelihood_ratio(events)
        pred["mld"], gaps["mld"] = llr < 0, np.abs(llr)
        extras["mld_llr"] = llr
        pred["mwpm"], extras["mwpm_weight"] = self.mwpm.predict_with_weights(
            events)
        pred["mwpm_corr"] = self.mwpm_corr.predict(events)
        for name in ("mwe", "mwpm_gap", "mwpm_classaware"):
            weights = getattr(self, name).class_weights(events)
            extras[name + "_weights"] = weights
            pred[name], gaps[name] = decision_from_class_weights(weights)
        for name in ("mwpm_obs", "mwpm_obs_corr"):
            weights = getattr(self, name).class_weights(events)
            extras[name + "_weights"] = weights
            pred[name], gaps[name] = decision_from_class_weights(weights)
        return pred, gaps, extras


# ---------------------------------------------------------------------------
# Single-fault (first-order) analysis
# ---------------------------------------------------------------------------

def single_fault_table(suite: DecoderSuite) -> dict[str, np.ndarray]:
    """Decode the syndrome of every undecomposed DEM mechanism on its own.

    The first-order logical error rate of decoder ``D`` is
    ``sum_i p_i [D(s_i) != l_i]`` (exact up to O(p^2)).

    Returns:
        ``probability``, ``flips`` (true logical flip of each mechanism),
        ``size`` (number of detectors), ``wrong_<decoder>`` booleans and
        ``gap_<decoder>`` for the soft decoders.
    """
    dem = suite.lrb_circuit.circuit.detector_error_model()
    mechs = [m for m in M.dem_mechanisms(dem) if m.probability > 0]
    n = dem.num_detectors
    events = np.zeros((len(mechs), n), dtype=bool)
    for i, m in enumerate(mechs):
        events[i, list(m.detectors)] = True
    flips = np.array([m.flips_observable for m in mechs])
    pred, gaps, extras = suite.run(events)
    table = {"probability": np.array([m.probability for m in mechs]),
             "flips": flips,
             "size": np.array([len(m.detectors) for m in mechs]),
             "mld_tie": np.abs(extras["mld_llr"]) < TIE_TOL}
    for name, prediction in pred.items():
        table["wrong_" + name] = prediction != flips
    for name, gap in gaps.items():
        table["gap_" + name] = gap
    return table


def pymatching_l_attribution(dem_decomposed: stim.DetectorErrorModel
                             ) -> dict[tuple[tuple[int, ...], bool], str]:
    """Classify each decomposed line by how PyMatching sees its observable.

    Keys are the line's total ``(detectors, flips_L)``; values are one of
    ``"undetectable_remainder"`` (a zero-detector component was dropped),
    ``"kept_observable_conflict"`` (some component's observable differs from
    the one PyMatching kept for that edge), ``"consistent"``. A symptom with
    several decomposed lines gets the worst label.
    """
    merge = pymatching_merge_model(dem_decomposed)
    rank = {"consistent": 0, "kept_observable_conflict": 1,
            "undetectable_remainder": 2}
    out: dict[tuple[tuple[int, ...], bool], str] = {}
    for instruction in dem_decomposed.flattened():
        if instruction.type != "error":
            continue
        comps = error_components(instruction)
        total, mask = _symmetric_difference(comps)
        key = (total, bool(mask & 1))
        if any(not d for d, _ in comps):
            label = "undetectable_remainder"
        elif any(merge[d].kept_flips != bool(m & 1) for d, m in comps):
            label = "kept_observable_conflict"
        else:
            label = "consistent"
        if key not in out or rank[label] > rank[out[key]]:
            out[key] = label
    return out
