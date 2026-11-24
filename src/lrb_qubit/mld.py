"""Exact maximum-likelihood decoding of LRB circuits by a time-slot transfer matrix.

For the circuits built by ``circuits.build_lrb_circuit`` every detector belongs
to a time slot (one check round, or the final data readout), and every error
mechanism of the detector error model (DEM) touches detectors in at most two
consecutive slots. The joint probability ``P(syndrome, L)`` of the full
detection-event vector and the logical flip ``L`` can therefore be computed
exactly by sweeping slot by slot while carrying a distribution over the
current slot's still-unconditioned detector bits and ``L`` (at most
``2**(4+1) = 32`` numbers per shot).

Let ``a`` be the bits of slot ``t-1``, ``b`` those of slot ``t`` and ``l`` the
logical bit. Mechanisms whose latest slot is ``t`` are convolved (over
``Z_2``) into a fixed kernel ``K_t(a, b, l)``. With ``psi_{t-1}(a, l)`` the
distribution before slot ``t``'s mechanisms and ``s`` the observed bits of slot
``t-1``,

    psi_t(b, l) = sum_{c, l'} psi_{t-1}(s XOR c, l') K_t(c, b, l XOR l').

The XOR re-indexing is a per-shot gather and the remaining sum is one matrix
product with a shot-independent ``32 x 32`` matrix. After the last slot,
``P(L = l | syndrome)`` is proportional to ``psi_T(s_T, l)``.

The DEM is exact for this noise model (Stim decomposes the depolarizing
channels into independent Pauli components exactly), so the decoder is the
exact maximum-likelihood decoder, and ``posterior`` is the exact soft output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import stim


@dataclass(frozen=True)
class Mechanism:
    probability: float
    detectors: tuple[int, ...]
    flips_observable: bool


def dem_mechanisms(dem: stim.DetectorErrorModel,
                   observable: int = 0) -> list[Mechanism]:
    """Error mechanisms of a DEM (no decomposition separators expected)."""
    mechanisms = []
    for instruction in dem.flattened():
        if instruction.type != "error":
            continue
        probability = instruction.args_copy()[0]
        detectors = []
        flips = False
        for target in instruction.targets_copy():
            if target.is_separator():
                continue
            if target.is_relative_detector_id():
                detectors.append(target.val)
            elif target.is_logical_observable_id():
                if target.val == observable:
                    flips = not flips
        mechanisms.append(Mechanism(probability, tuple(sorted(detectors)),
                                    flips))
    return mechanisms


def lrb_detector_slots(lrb_circuit) -> np.ndarray:
    """Slot index of every detector of an ``LRBCircuit``."""
    slots = []
    final = lrb_circuit.depth + 1
    for round_label, _ in lrb_circuit.detector_keys:
        if round_label == final:
            slots.append(lrb_circuit.num_rounds)
        else:
            slots.append(lrb_circuit.rounds.index(round_label))
    return np.asarray(slots, dtype=np.int64)


def _convolve(kernel: np.ndarray, vector: int, probability: float) -> None:
    """In-place Z_2 convolution with a two-point distribution {0: 1-p, v: p}."""
    if probability <= 0.0 or vector == 0:
        return
    index = np.arange(kernel.size) ^ vector
    kernel[:] = (1.0 - probability) * kernel + probability * kernel[index]


class TransferMatrixMLD:
    """Exact MLD over a time-sliced DEM.

    Args:
        dem: Detector error model with observable 0 as the logical.
        detector_slots: Slot index for each detector (non-decreasing in time).
    """

    def __init__(self, dem: stim.DetectorErrorModel,
                 detector_slots: Sequence[int]):
        slots = np.asarray(detector_slots, dtype=np.int64)
        if len(slots) != dem.num_detectors:
            raise ValueError("Need one slot per detector.")
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
            n_cur = self.slot_sizes[t]
            kernel = np.zeros(2 ** (n_prev + n_cur + 1))
            kernel[0] = 1.0
            kernels.append(kernel)

        self.num_mechanisms = 0
        # Undetectable logical mechanisms (no detectors, observable flipped)
        # occur e.g. with the two-qubit terminal readout. They shift every
        # posterior by the same amount, so they are folded into slot 0.
        self.undetectable_logical_probability = 0.0
        for mech in dem_mechanisms(dem):
            if not mech.detectors:
                if mech.flips_observable and mech.probability > 0:
                    q = self.undetectable_logical_probability
                    self.undetectable_logical_probability = (
                        q + mech.probability - 2 * q * mech.probability)
                    _convolve(kernels[0], 1 << self.slot_sizes[0],
                              mech.probability)
                    self.num_mechanisms += 1
                continue
            mech_slots = slots[list(mech.detectors)]
            t = int(mech_slots.max())
            if int(mech_slots.min()) < t - 1:
                raise ValueError(
                    "Mechanism spans non-adjacent slots: "
                    f"{sorted(set(mech_slots.tolist()))}")
            n_prev = self.slot_sizes[t - 1] if t > 0 else 0
            n_cur = self.slot_sizes[t]
            vector = 0
            for det in mech.detectors:
                bit = int(position[det])
                if slots[det] == t:
                    vector ^= 1 << (n_prev + bit)
                else:
                    vector ^= 1 << bit
            if mech.flips_observable:
                vector ^= 1 << (n_prev + n_cur)
            _convolve(kernels[t], vector, mech.probability)
            self.num_mechanisms += 1

        # Initial state psi_0(b, l) = K_0(b, l).
        self.initial = kernels[0].copy()
        # Transfer matrices: T_t[(c, l'), (b, l)] = K_t(c, b, l XOR l').
        self.transfer = [None]
        for t in range(1, self.num_slots):
            n_prev = self.slot_sizes[t - 1]
            n_cur = self.slot_sizes[t]
            kernel = kernels[t].reshape(2, 2 ** n_cur, 2 ** n_prev)
            # kernel[l, b, c] = K_t(c, b, l)
            matrix = np.zeros((2 * 2 ** n_prev, 2 * 2 ** n_cur))
            for l_in in range(2):
                for l_out in range(2):
                    block = kernel[l_in ^ l_out].T  # (c, b)
                    matrix[l_in * 2 ** n_prev:(l_in + 1) * 2 ** n_prev,
                           l_out * 2 ** n_cur:(l_out + 1) * 2 ** n_cur] = block
            self.transfer.append(matrix)

    def _slot_values(self, detection_events: np.ndarray, t: int) -> np.ndarray:
        dets = self.slot_detectors[t]
        weights = (1 << np.arange(len(dets))).astype(np.int64)
        return detection_events[:, dets].astype(np.int64) @ weights

    def log_likelihood_ratio(self, detection_events: np.ndarray,
                             batch_size: int = 65536) -> np.ndarray:
        """``log P(L=0 | s) - log P(L=1 | s)`` for each shot."""
        detection_events = np.asarray(detection_events, dtype=bool)
        out = np.empty(detection_events.shape[0])
        for start in range(0, detection_events.shape[0], batch_size):
            chunk = detection_events[start:start + batch_size]
            out[start:start + len(chunk)] = self._llr_chunk(chunk)
        return out

    def _llr_chunk(self, events: np.ndarray) -> np.ndarray:
        shots = events.shape[0]
        psi = np.broadcast_to(self.initial, (shots, self.initial.size)).copy()
        for t in range(1, self.num_slots):
            n_prev = self.slot_sizes[t - 1]
            s = self._slot_values(events, t - 1)
            size = 2 ** n_prev
            base = np.arange(2 * size)
            cols = (base[None, :] & (size - 1)) ^ s[:, None]
            cols |= (base[None, :] >> n_prev) << n_prev
            phi = np.take_along_axis(psi, cols, axis=1)
            psi = phi @ self.transfer[t]
            psi /= psi.max(axis=1, keepdims=True)
        n_last = self.slot_sizes[-1]
        s = self._slot_values(events, self.num_slots - 1)
        p0 = psi[np.arange(shots), s]
        p1 = psi[np.arange(shots), s + 2 ** n_last]
        with np.errstate(divide="ignore"):
            return np.log(p0) - np.log(p1)

    def decode(self, detection_events: np.ndarray) -> tuple[np.ndarray,
                                                            np.ndarray]:
        """Return ``(predicted_flip, gap)`` with ``gap = |LLR|``."""
        llr = self.log_likelihood_ratio(detection_events)
        return llr < 0, np.abs(llr)


def brute_force_joint(dem: stim.DetectorErrorModel) -> np.ndarray:
    """Exact ``P(detectors, L)`` as a dense vector (small DEMs only).

    Index ``i`` encodes detector ``k`` in bit ``k`` and ``L`` in the top bit.
    """
    n = dem.num_detectors
    if n > 22:
        raise ValueError("Too many detectors for brute force.")
    joint = np.zeros(2 ** (n + 1))
    joint[0] = 1.0
    for mech in dem_mechanisms(dem):
        vector = 0
        for det in mech.detectors:
            vector ^= 1 << det
        if mech.flips_observable:
            vector ^= 1 << n
        _convolve(joint, vector, mech.probability)
    return joint
