"""LRB circuits, decoders and counting for unrotated surface codes (d = 2, 3).

This generalizes ``lrb_qubit.circuits.build_lrb_circuit`` to any code object
with the interface of ``d3_code.UnrotatedSurfaceCode`` (or the adapter
``LeadFiveQubitCode`` around ``lrb_qubit.code``) and adds a measurement
schedule option:

``schedule="sequential"``
    The repo layout: each check is entangled and then measured on its own,
    so a round has one single-ancilla measurement layer per stabilizer (each
    puts ``DEPOLARIZE1(2p)`` on every other wire), then one reset layer.
    With ``LeadFiveQubitCode`` this reproduces ``build_lrb_circuit`` exactly
    (same Stim circuit text; tested).
``schedule="parallel"``
    The same gates in the same order, but all ancillas are measured in one
    layer at the end of the round (one ``DEPOLARIZE1(2p)`` idle layer instead
    of one per stabilizer).

The LRB structure is unchanged: ideal encoding of ``|+>_L``; for each forward
Clifford the logical gates, a noisy check round and an ancilla reset; the
inverses ``C_m^-1 ... C_1^-1`` with no checks; a final check round (no
reset); an ideal X readout of every data qubit. Noise is the repo's SI1000
model via ``lrb_qubit.circuits.emit_operations``. Detectors are the lead's
difference detectors (round ``r`` stabilizer ``s`` against the previous
round's generators that equal ``s`` after back-propagation through the
logical gates in between) plus final X-stabilizer detectors from the data
readout; observable 0 is the ``X_L`` readout parity.

Decoders: PyMatching on Stim's graphlike decomposition (optionally with
PyMatching's correlated matching), BP-OSD from ``ldpc`` 2.4.1 on the full
hypergraph DEM, and (d = 2 only) the lead's exact transfer-matrix MLD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np
import stim

from . import baseline as B
from . import circuits as Q
from . import code as C
from . import experiment as E

Operation = tuple[str, tuple[int, ...]]

SCHEDULES = ("sequential", "parallel")


# ----------------------------------------------------------------------
# Adapter: the lead's [[5,1,2]] module behind the generic interface
# ----------------------------------------------------------------------
class LeadFiveQubitCode:
    """``lrb_qubit.code`` exposed with the ``UnrotatedSurfaceCode`` interface.

    Every operation list comes from the lead's module, so circuits built from
    this adapter are the lead's circuits.
    """

    distance = 2

    def __init__(self) -> None:
        self.num_data = C.NUM_DATA
        self.num_qubits = C.NUM_QUBITS
        self.data_qubits = C.DATA_QUBITS
        self.stabilizer_paulis = C.STABILIZER_PAULIS
        self.stabilizer_names = C.STABILIZER_NAMES
        self.num_stabilizers = len(C.STABILIZER_PAULIS)
        self.z_stabilizers = C.Z_STABILIZERS
        self.x_stabilizers = C.X_STABILIZERS
        self.stabilizer_ancilla = C.STABILIZER_ANCILLA
        self.ancilla_qubits = C.ANCILLA_QUBITS
        self.logical_x = C.LOGICAL_X
        self.logical_z = C.LOGICAL_Z
        self.logical_x_support = tuple(q for q, ch in enumerate(C.LOGICAL_X)
                                       if ch != "_")
        blocks: dict[int, list[Operation]] = {}
        current: list[Operation] = []
        for gate, targets in (C.z_stabilizer_check_operations()
                              + C.x_stabilizer_check_operations()):
            if gate == "M":
                blocks[C.STABILIZER_ANCILLA.index(targets[0])] = current
                current = []
            else:
                current.append((gate, targets))
        self._checks = blocks

    def check_operations(self, index: int) -> list[Operation]:
        return list(self._checks[index])

    def encoding_operations(self) -> list[Operation]:
        return C.encoding_operations()

    def logical_clifford_operations(self, descriptor: str,
                                    inverse: bool = False) -> list[Operation]:
        return C.logical_clifford_operations(descriptor, inverse)

    def data_tableau(self, operations: Sequence[Operation]) -> stim.Tableau:
        return C.data_tableau(operations)

    def decompose_in_stabilizers(self, pauli: stim.PauliString
                                 ) -> tuple[int, ...]:
        return C.decompose_in_stabilizers(pauli)

    def stabilizer_pauli(self, index: int) -> stim.PauliString:
        return C.stabilizer_pauli(index)


# ----------------------------------------------------------------------
# Circuit construction
# ----------------------------------------------------------------------
def check_round_operations(code, schedule: str = "sequential",
                           reset: bool = True) -> list[Operation]:
    """One check round (Z checks, then X checks), optionally with reset."""
    if schedule not in SCHEDULES:
        raise ValueError(f"schedule must be one of {SCHEDULES}.")
    order = list(code.z_stabilizers) + list(code.x_stabilizers)
    ops: list[Operation] = []
    if schedule == "sequential":
        for s in order:
            ops += code.check_operations(s)
            ops.append(("M", (code.stabilizer_ancilla[s],)))
    else:
        for s in order:
            ops += code.check_operations(s)
        ops += [("M", (code.stabilizer_ancilla[s],)) for s in order]
    if reset:
        ops += [("R", (q,)) for q in code.ancilla_qubits]
    return ops


@dataclass
class GenericLRBCircuit:
    """Stim LRB circuit plus bookkeeping (generalizes ``LRBCircuit``).

    ``records`` maps each round label to the measurement records of the
    stabilizers in stabilizer order (index ``s``), whatever the schedule.
    """

    circuit: stim.Circuit
    depth: int
    rounds: list[int]
    detector_keys: list[tuple[int, int]]
    back: dict[int, tuple[tuple[int, ...], ...]]
    num_stabilizers: int
    schedule: str
    segments: list[list[Operation]] = field(default_factory=list)
    num_final_detectors: int = 0

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    @property
    def num_round_detectors(self) -> int:
        return self.num_stabilizers * self.num_rounds

    def raw_syndromes(self, detection_events: np.ndarray) -> np.ndarray:
        """``(shots, rounds, S)`` raw stabilizer flips (see ``LRBCircuit``)."""
        S = self.num_stabilizers
        shots = detection_events.shape[0]
        raw = np.zeros((shots, self.num_rounds, S), dtype=bool)
        for k, label in enumerate(self.rounds):
            det = detection_events[:, S * k:S * k + S]
            raw[:, k, :] = det
            if k > 0:
                for s in range(S):
                    for prev_s in self.back[label][s]:
                        raw[:, k, s] ^= raw[:, k - 1, prev_s]
        return raw


def _back_subsets(code, segment: Sequence[Operation]
                  ) -> tuple[tuple[int, ...], ...]:
    inverse = code.data_tableau(segment).inverse()
    return tuple(code.decompose_in_stabilizers(
        inverse(code.stabilizer_pauli(s))) for s in range(code.num_stabilizers))


def build_lrb_circuit(code,
                      cliffords: Sequence[str],
                      depth: int,
                      p: float,
                      *,
                      schedule: str = "sequential",
                      noise: Optional[Q.SI1000] = Q.SI1000(),
                      terminal_noise: bool = False,
                      forward_check_rounds: Optional[Iterable[int]] = None,
                      terminal_check: bool = True,
                      inverse_check_every: int = 0) -> GenericLRBCircuit:
    """Encoded LRB circuit of depth ``depth`` (full X readout of the data).

    Arguments mirror ``lrb_qubit.circuits.build_lrb_circuit`` with
    ``terminal="data5"`` semantics (all data read out in the X basis),
    plus ``schedule``.
    """
    if len(cliffords) < depth:
        raise ValueError("Clifford sequence is shorter than the depth.")
    forward = list(cliffords[:depth])
    forward_set = (set(range(depth)) if forward_check_rounds is None
                   else {int(r) for r in forward_check_rounds})
    if any(r < 0 or r >= depth for r in forward_set):
        raise ValueError("forward_check_rounds must lie in [0, depth).")
    n_q = code.num_qubits
    S = code.num_stabilizers

    circuit = stim.Circuit()
    Q.emit_operations(circuit, code.encoding_operations(), p, None, n_q)

    rounds: list[int] = []
    round_records: dict[int, list[int]] = {}
    segments: list[list[Operation]] = []
    pending: list[Operation] = []
    order = list(code.z_stabilizers) + list(code.x_stabilizers)

    def run_check_round(label: int, reset: bool = True) -> None:
        ops = check_round_operations(code, schedule, reset=reset)
        records = Q.emit_operations(circuit, ops, p, noise, n_q,
                                    noisy_measurements=True)
        by_stabilizer = [0] * S
        for s, record in zip(order, records):
            by_stabilizer[s] = record
        rounds.append(label)
        round_records[label] = by_stabilizer
        segments.append(list(pending))
        pending.clear()

    for j, descriptor in enumerate(forward):
        ops = code.logical_clifford_operations(descriptor)
        Q.emit_operations(circuit, ops, p, noise, n_q,
                          noisy_measurements=False)
        pending.extend(ops)
        if j in forward_set:
            run_check_round(j)

    inverse_label = Q.INVERSE_ROUND_OFFSET
    for j, descriptor in enumerate(reversed(forward)):
        ops = code.logical_clifford_operations(descriptor, inverse=True)
        Q.emit_operations(circuit, ops, p, noise, n_q,
                          noisy_measurements=False)
        pending.extend(ops)
        if (inverse_check_every > 0 and (j + 1) % inverse_check_every == 0
                and j + 1 < depth):
            run_check_round(inverse_label)
            inverse_label += 1

    if terminal_check:
        run_check_round(depth, reset=False)
    segments.append(list(pending))
    pending.clear()

    readout_ops = ([("H", (q,)) for q in code.data_qubits]
                   + [("M", (q,)) for q in code.data_qubits])
    readout_records = Q.emit_operations(
        circuit, readout_ops, p, noise if terminal_noise else None, n_q,
        noisy_measurements=True)
    data_record = dict(zip(code.data_qubits, readout_records))

    detector_keys: list[tuple[int, int]] = []
    back: dict[int, tuple[tuple[int, ...], ...]] = {}
    previous: Optional[list[int]] = None
    for position, label in enumerate(rounds):
        records = round_records[label]
        if previous is None:
            subsets = tuple(() for _ in range(S))
        else:
            subsets = _back_subsets(code, segments[position])
        back[label] = subsets
        for s in range(S):
            targets = [records[s]]
            if previous is not None:
                targets += [previous[t] for t in subsets[s]]
            circuit.append(
                "DETECTOR",
                [stim.target_rec(r - circuit.num_measurements) for r in targets],
                [s, label])
            detector_keys.append((label, s))
        previous = records

    final_subsets = _back_subsets(code, segments[-1])
    for s in code.x_stabilizers:
        support = [q for q, ch in enumerate(code.stabilizer_paulis[s])
                   if ch != "_"]
        targets = [data_record[q] for q in support]
        if previous is not None:
            targets += [previous[t] for t in final_subsets[s]]
        circuit.append(
            "DETECTOR",
            [stim.target_rec(r - circuit.num_measurements) for r in targets],
            [s, depth + 1])
        detector_keys.append((depth + 1, s))

    circuit.append(
        "OBSERVABLE_INCLUDE",
        [stim.target_rec(data_record[q] - circuit.num_measurements)
         for q in code.logical_x_support], 0)

    return GenericLRBCircuit(circuit=circuit, depth=depth, rounds=rounds,
                             detector_keys=detector_keys, back=back,
                             num_stabilizers=S, schedule=schedule,
                             segments=segments,
                             num_final_detectors=len(code.x_stabilizers))


def build_lrb_const0_circuit(code, cliffords: Sequence[str], depth: int,
                             p: float, *,
                             noise: Optional[Q.SI1000] = Q.SI1000()
                             ) -> GenericLRBCircuit:
    """``const=0``: no check rounds, ideal X readout of all data."""
    return build_lrb_circuit(code, cliffords, depth, p, noise=noise,
                             forward_check_rounds=(), terminal_check=False)


# ----------------------------------------------------------------------
# Physical RB (exact)
# ----------------------------------------------------------------------
def clifford_gate_count(descriptor: str) -> int:
    """Physical gates of one Clifford in ``build_physical_rb_circuit``."""
    return len(C.clifford_gate_sequence(descriptor))


def physical_rb_fidelity(sequence: Sequence[str], depth: int, p_rb: float,
                         noise: Q.SI1000 = Q.SI1000()) -> float:
    """Exact ``F = |P(0) - P(1)|`` of ``build_physical_rb_circuit``.

    Every gate is followed by ``DEPOLARIZE1(q)`` with ``q = 0.1 p_rb``. The
    depolarizing channel commutes with every unitary, so after ``N`` gates
    the Bloch vector is shrunk by ``(1 - 4q/3)^N``; the forward and inverse
    halves have the same gate count.
    """
    q = noise.one_qubit * p_rb
    gates = 2 * sum(clifford_gate_count(c) for c in sequence[:depth])
    return (1.0 - 4.0 * q / 3.0) ** gates


def physical_rb_decay(p_rb: float, noise: Q.SI1000 = Q.SI1000()) -> float:
    """Ensemble decay per depth step for uniformly random Cliffords.

    ``f = mean_C (1 - 4q/3)^(2 len(C))`` over the 24-element catalog, so the
    sequence-averaged fidelity is exactly ``f^m``.
    """
    q = noise.one_qubit * p_rb
    lengths = np.array([clifford_gate_count(c)
                        for c in C.SINGLE_QUBIT_CLIFFORDS])
    return float(np.mean((1.0 - 4.0 * q / 3.0) ** (2 * lengths)))


# ----------------------------------------------------------------------
# Decoders
# ----------------------------------------------------------------------
class MatchingDecoder:
    """PyMatching on Stim's graphlike decomposition of the DEM.

    Attributes:
        decomposition_failures: True when Stim could not decompose every
            hyperedge (then ``ignore_decomposition_failures`` was used and
            PyMatching drops those mechanisms).
    """

    def __init__(self, circuit: stim.Circuit, correlated: bool = False):
        import pymatching
        self.decomposition_failures = False
        try:
            dem = circuit.detector_error_model(decompose_errors=True)
        except ValueError:
            dem = circuit.detector_error_model(
                decompose_errors=True, ignore_decomposition_failures=True)
            self.decomposition_failures = True
        self.dem = dem
        self.correlated = correlated
        self.matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=correlated)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        if detection_events.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        pred = self.matching.decode_batch(
            np.asarray(detection_events, dtype=np.uint8),
            enable_correlations=self.correlated)
        return np.asarray(pred[:, 0], dtype=bool)


def dem_matrices(dem: stim.DetectorErrorModel, observable: int = 0):
    """Check matrix, observable row and priors of a DEM (merged columns).

    Mechanisms with identical (detectors, observable) are merged as
    independent errors. Returns ``(H, L, priors)`` with ``H`` a CSC
    matrix of shape ``(num_detectors, num_columns)``.
    """
    import scipy.sparse as sp
    merged: dict[tuple[tuple[int, ...], bool], float] = {}
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        prob = instruction.args_copy()[0]
        dets: set[int] = set()
        flip = False
        for target in instruction.targets_copy():
            if target.is_separator():
                continue
            if target.is_relative_detector_id():
                dets ^= {target.val}
            elif target.is_logical_observable_id() and target.val == observable:
                flip = not flip
        key = (tuple(sorted(dets)), flip)
        if not key[0] and not flip:
            continue
        old = merged.get(key, 0.0)
        merged[key] = old + prob - 2 * old * prob
    keys = list(merged)
    rows, cols = [], []
    for j, (dets, _) in enumerate(keys):
        rows += list(dets)
        cols += [j] * len(dets)
    H = sp.csc_matrix((np.ones(len(rows), dtype=np.uint8), (rows, cols)),
                      shape=(dem.num_detectors, len(keys)))
    L = np.array([flip for _, flip in keys], dtype=np.uint8)
    priors = np.array([merged[k] for k in keys])
    return H, L, priors


class BpOsdDEMDecoder:
    """BP-OSD (``ldpc.BpOsdDecoder``) on the full hypergraph DEM.

    Product-sum BP (``max_iter`` iterations) followed by OSD-CS of order
    ``osd_order``. ``ldpc`` skips OSD whenever BP's hard decision satisfies
    the syndrome, and on these circuits BP sometimes "converges" to a valid
    but absurd solution (5-13 mechanisms with prior ~1e-28) even for a
    single-fault syndrome. When BP converged, the decoder therefore also
    tries (a) BP with ``fallback_iter`` iterations, after which OSD almost
    always runs, and (b) the most likely single mechanism with exactly this
    syndrome, and keeps the candidate with the largest prior probability
    (smallest ``sum log((1-p)/p)``). Each distinct syndrome in a batch is
    decoded once.
    """

    def __init__(self, circuit: stim.Circuit, *, max_iter: int = 30,
                 fallback_iter: int = 2, bp_method: str = "product_sum",
                 osd_method: str = "osd_cs", osd_order: int = 10):
        from ldpc import BpOsdDecoder
        dem = circuit.detector_error_model(decompose_errors=False)
        H, L, priors = dem_matrices(dem)
        keep = np.asarray(H.sum(axis=0)).ravel() > 0
        self.H = H[:, keep]
        self.L = L[keep].astype(np.int64)
        self.priors = priors[keep]
        self.weights = np.log((1.0 - self.priors) / self.priors)
        self.num_detectors = dem.num_detectors
        kwargs = dict(error_channel=list(self.priors), bp_method=bp_method,
                      osd_method=osd_method, osd_order=osd_order)
        self.decoder = BpOsdDecoder(self.H, max_iter=max_iter, **kwargs)
        self.fallback = (BpOsdDecoder(self.H, max_iter=fallback_iter, **kwargs)
                         if fallback_iter else None)
        # Most likely single mechanism for each single-fault syndrome.
        self.single: dict[bytes, tuple[float, int]] = {}
        dense = self.H.tocsc()
        for j in range(dense.shape[1]):
            rows = dense.indices[dense.indptr[j]:dense.indptr[j + 1]]
            syndrome = np.zeros(self.num_detectors, dtype=np.uint8)
            syndrome[rows] = 1
            key = np.packbits(syndrome).tobytes()
            if key not in self.single or self.weights[j] < self.single[key][0]:
                self.single[key] = (float(self.weights[j]), int(self.L[j]))
        self.num_fallbacks = 0

    def _decode_one(self, syndrome: np.ndarray) -> bool:
        error = self.decoder.decode(syndrome).astype(bool)
        best_w = float(self.weights[error].sum())
        best_l = int(self.L[error].sum()) % 2
        if self.decoder.converge:
            candidates = []
            if self.fallback is not None:
                other = self.fallback.decode(syndrome).astype(bool)
                if not np.any((self.H @ other) % 2 != syndrome):
                    candidates.append((float(self.weights[other].sum()),
                                       int(self.L[other].sum()) % 2))
            single = self.single.get(np.packbits(syndrome).tobytes())
            if single is not None:
                candidates.append(single)
            for w, l in candidates:
                if w < best_w - 1e-9:
                    best_w, best_l = w, l
                    self.num_fallbacks += 1
        return bool(best_l)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        dets = np.asarray(detection_events, dtype=np.uint8)
        out = np.zeros(dets.shape[0], dtype=bool)
        if dets.shape[0] == 0:
            return out
        packed = np.packbits(dets, axis=1)
        unique, inverse = np.unique(packed, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).ravel()
        predictions = np.zeros(len(unique), dtype=bool)
        for u in range(len(unique)):
            syndrome = np.unpackbits(unique[u])[:self.num_detectors]
            if not syndrome.any():
                continue
            predictions[u] = self._decode_one(syndrome)
        out[:] = predictions[inverse]
        return out


class BruteForceMLD:
    """Exact MLD by enumerating all syndromes (DEMs with <= 20 detectors).

    Used for the ``const=0`` circuits, whose only detectors are the final
    X-stabilizer readouts (6 at d=3, 2 at d=2).
    """

    def __init__(self, circuit: stim.Circuit):
        from . import mld as M
        dem = circuit.detector_error_model(decompose_errors=False)
        self.n = dem.num_detectors
        joint = M.brute_force_joint(dem)
        size = 2 ** self.n
        self.flip = joint[size:] > joint[:size]

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        if self.n == 0:
            return np.zeros(detection_events.shape[0], dtype=bool)
        weights = (1 << np.arange(self.n)).astype(np.int64)
        index = np.asarray(detection_events, dtype=np.int64) @ weights
        return self.flip[index]


def logical_frame(code, pauli: stim.PauliString) -> Optional[str]:
    """``"X"``, ``"Y"`` or ``"Z"`` if ``pauli`` is that logical times a
    stabilizer (any sign), else ``None``."""
    x = stim.PauliString(code.logical_x)
    z = stim.PauliString(code.logical_z)
    for name, logical in (("X", x), ("Z", z), ("Y", x * z)):
        try:
            code.decompose_in_stabilizers(pauli * logical)
        except ValueError:
            continue
        return name
    return None


def observable_frames(code, lc: GenericLRBCircuit) -> dict[int, str]:
    """Logical frame of the final ``X_L`` observable at each check round.

    ``frames[label]`` is the class (``X``, ``Y`` or ``Z``) of ``U^dag X_L U``
    where ``U`` is the product of all logical gates after that round. Data
    errors during that round flip the observable iff they anticommute with
    this operator.
    """
    frames: dict[int, str] = {}
    after: list[Operation] = []
    for position in range(lc.num_rounds - 1, -1, -1):
        after = list(lc.segments[position + 1]) + after
        tableau = code.data_tableau(after).inverse()
        frame = logical_frame(code, tableau(stim.PauliString(code.logical_x)))
        if frame is None:
            raise ValueError("Observable is not a logical operator.")
        frames[lc.rounds[position]] = frame
    return frames


def observable_detector_mask(code, lc: GenericLRBCircuit) -> np.ndarray:
    """Detectors kept by observable-frame matching.

    In a round whose frame is ``X_L`` only X-check detectors are kept (only
    Z components of errors can flip the observable), in a ``Z_L`` frame only
    Z-check detectors, in a ``Y_L`` frame both. The final data detectors
    (frame ``X_L``) are kept.
    """
    frames = observable_frames(code, lc)
    z_set = set(code.z_stabilizers)
    keep = []
    for label, s in lc.detector_keys:
        if label == lc.depth + 1:
            keep.append(True)
            continue
        frame = frames[label]
        kind = "Z" if s in z_set else "X"
        keep.append(frame == "Y" or frame == kind)
    return np.asarray(keep, dtype=bool)


def restrict_detectors(circuit: stim.Circuit, keep: np.ndarray
                       ) -> stim.Circuit:
    """Copy of ``circuit`` with only the detectors flagged in ``keep``."""
    out = stim.Circuit()
    index = 0
    for instruction in circuit.flattened():
        if instruction.name == "DETECTOR":
            if keep[index]:
                out.append(instruction)
            index += 1
        else:
            out.append(instruction)
    if index != len(keep):
        raise ValueError("keep mask does not match the detector count.")
    return out


class ObservableMatchingDecoder:
    """PyMatching on the observable-frame subgraph of the DEM.

    A single-observable form of the logical-observable matching idea of
    Serra-Peralta, Shaw and Terhal (arXiv:2505.13599): detectors that only
    see error components that cannot flip the final ``X_L`` are dropped
    (``observable_detector_mask``), which removes the three-detector
    hyperedges that the fold-transversal S creates between X- and Z-type
    detectors. Stim then decomposes the remaining Y-type hyperedges.
    """

    def __init__(self, code, lc: GenericLRBCircuit,
                 correlated: bool = False):
        import pymatching
        self.keep = observable_detector_mask(code, lc)
        self.circuit = restrict_detectors(lc.circuit, self.keep)
        self.decomposition_failures = False
        try:
            dem = self.circuit.detector_error_model(decompose_errors=True)
        except ValueError:
            dem = self.circuit.detector_error_model(
                decompose_errors=True, ignore_decomposition_failures=True)
            self.decomposition_failures = True
        self.dem = dem
        self.correlated = correlated
        self.matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=correlated)

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        if detection_events.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        pred = self.matching.decode_batch(
            np.asarray(detection_events[:, self.keep], dtype=np.uint8),
            enable_correlations=self.correlated)
        return np.asarray(pred[:, 0], dtype=bool)


class MLDDecoder:
    """The lead's exact transfer-matrix MLD (slots of at most 8 detectors)."""

    def __init__(self, lc: GenericLRBCircuit):
        from . import mld as M
        slots = []
        final = lc.depth + 1
        for label, _ in lc.detector_keys:
            slots.append(lc.num_rounds if label == final
                         else lc.rounds.index(label))
        self.decoder = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                           np.asarray(slots))

    def predict(self, detection_events: np.ndarray) -> np.ndarray:
        if detection_events.shape[0] == 0:
            return np.zeros(0, dtype=bool)
        return self.decoder.log_likelihood_ratio(detection_events) < 0


def single_fault_report(circuit: stim.Circuit, decoder) -> dict:
    """Decode every single DEM mechanism; list those decoded wrongly.

    A decoder that corrects all single faults has failures only at second
    order in p. Returns the number of mechanisms, the failing ones and their
    total probability (the first-order coefficient of the logical error).
    """
    dem = circuit.detector_error_model(decompose_errors=False)
    H, L, priors = dem_matrices(dem)
    dense = H.toarray().T.astype(np.uint8)
    pred = decoder.predict(dense)
    wrong = pred != L.astype(bool)
    failing = [
        {"detectors": [int(x) for x in np.flatnonzero(dense[j])],
         "flips_observable": bool(L[j]), "probability": float(priors[j])}
        for j in np.flatnonzero(wrong)]
    return {"num_mechanisms": int(len(L)), "num_failing": int(wrong.sum()),
            "failing_probability": float(priors[wrong].sum()),
            "failing": failing}


# ----------------------------------------------------------------------
# Counting
# ----------------------------------------------------------------------
def run_decoder_depth(code, schedule: str, p: float, sequence: Sequence[str],
                      depth: int, shots: int, *, decoder: str = "bposd",
                      seed: int = 0, time_budget: float = float("inf"),
                      chunk: int = 25,
                      noise: Q.SI1000 = Q.SI1000()) -> dict:
    """Decode up to ``shots`` fresh shots of one circuit with one decoder.

    Stops early (after a whole chunk) once ``time_budget`` seconds have
    passed, so slow decoders give a known, saved number of shots.

    Returns:
        ``{"shots", "errors", "seconds"}`` (corrected LRB: no rejection).
    """
    import time
    start = time.perf_counter()
    lc = build_lrb_circuit(code, sequence, depth, p, schedule=schedule,
                           noise=noise)
    if decoder == "bposd":
        dec = BpOsdDEMDecoder(lc.circuit)
    elif decoder == "pm":
        dec = MatchingDecoder(lc.circuit)
    elif decoder == "pmf":
        dec = ObservableMatchingDecoder(code, lc)
    elif decoder == "mld":
        dec = MLDDecoder(lc)
    else:
        raise ValueError(f"Unknown decoder {decoder!r}.")
    sampler = lc.circuit.compile_detector_sampler(seed=seed)
    done = errors = 0
    while done < shots and time.perf_counter() - start < time_budget:
        n = min(chunk, shots - done)
        dets, obs = sampler.sample(n, separate_observables=True)
        errors += int((dec.predict(dets) != obs[:, 0]).sum())
        done += n
    return {"shots": done, "errors": errors,
            "seconds": time.perf_counter() - start}


DEFAULT_POLICIES = ([(f"lrbd_const_{k}", "const", k) for k in range(1, 23)]
                    + [(f"lrbd_unif_{k}", "unif", k) for k in range(1, 23)])


def run_sequence(code, schedule: str, p: float, sequence: Sequence[str],
                 depths: Sequence[int], shots: int, *,
                 decoders: Sequence[str] = ("pm",),
                 decode_shots: Optional[dict[str, int]] = None,
                 policies=DEFAULT_POLICIES,
                 final_variant_policies=(("unif", 1), ("const", 1)),
                 const0: bool = True,
                 batch_size: int = 50_000,
                 seed: int = 0,
                 noise: Q.SI1000 = Q.SI1000()) -> E.PointResult:
    """All methods for one Clifford sequence at one ``p`` (one-sequence table).

    Methods: ``lrbd_*`` (LRB-D on raw check syndromes, ``data2`` semantics),
    ``lrbd_*_final`` (also requiring clean final data detectors), one table
    per decoder in ``decoders`` (``pm``, ``pmc``, ``bposd``, ``mld``), and
    ``const0`` / ``const0_mld`` (exact) from the separate no-check circuit.

    Args:
        decode_shots: Per-decoder cap on decoded shots per circuit (default:
            all ``shots``).
    """
    depths = list(depths)
    result = E.PointResult(p, depths, 1)
    decode_shots = dict(decode_shots or {})
    for d_index, depth in enumerate(depths):
        lc = build_lrb_circuit(code, sequence, depth, p, schedule=schedule,
                               noise=noise)
        built = {}
        for name in decoders:
            if name == "pm":
                built[name] = MatchingDecoder(lc.circuit)
            elif name == "pmc":
                built[name] = MatchingDecoder(lc.circuit, correlated=True)
            elif name == "pmf":
                built[name] = ObservableMatchingDecoder(code, lc)
            elif name == "bposd":
                built[name] = BpOsdDEMDecoder(lc.circuit)
            elif name == "mld":
                built[name] = MLDDecoder(lc)
            else:
                raise ValueError(f"Unknown decoder {name!r}.")
        n_round = lc.num_round_detectors
        sampler = lc.circuit.compile_detector_sampler(
            seed=seed + 1000 * d_index)
        done = 0
        while done < shots:
            n = min(batch_size, shots - done)
            dets, obs = sampler.sample(n, separate_observables=True)
            flip = obs[:, 0]
            raw = lc.raw_syndromes(dets)
            # Same as B.accept_mask, with the per-round reduction done once.
            round_clean = ~raw.any(axis=2)
            final_clean = ~dets[:, n_round:].any(axis=1)
            for name, policy, k in policies:
                used = B.used_rounds(policy, k, depth)
                columns = [lc.rounds.index(r) for r in used]
                acc = round_clean[:, columns].all(axis=1)
                result.table(name).add(0, d_index, n, int(acc.sum()),
                                       int((acc & flip).sum()))
                if (policy, k) in final_variant_policies:
                    acc_f = acc & final_clean
                    result.table(name + "_final").add(
                        0, d_index, n, int(acc_f.sum()),
                        int((acc_f & flip).sum()))
            for name, decoder in built.items():
                cap = decode_shots.get(name, shots)
                n_dec = min(n, max(cap - done, 0))
                if n_dec <= 0:
                    continue
                pred = decoder.predict(dets[:n_dec])
                wrong = pred != flip[:n_dec]
                result.table(name).add(0, d_index, n_dec, n_dec,
                                       int(wrong.sum()))
            done += n

        if const0:
            lc0 = build_lrb_const0_circuit(code, sequence, depth, p,
                                           noise=noise)
            dec0 = BruteForceMLD(lc0.circuit)
            sampler0 = lc0.circuit.compile_detector_sampler(
                seed=seed + 7 + 1000 * d_index)
            done = 0
            while done < shots:
                n = min(batch_size, shots - done)
                dets, obs = sampler0.sample(n, separate_observables=True)
                flip = obs[:, 0]
                acc = ~dets.any(axis=1)
                result.table("const0").add(0, d_index, n, int(acc.sum()),
                                           int((acc & flip).sum()))
                wrong = dec0.predict(dets) != flip
                result.table("const0_mld").add(0, d_index, n, n,
                                               int(wrong.sum()))
                done += n
    return result
