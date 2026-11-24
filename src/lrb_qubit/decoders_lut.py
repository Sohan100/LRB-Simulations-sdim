"""Sliding-window lookup-table (LUT) decoder with a running frame, built from exact window marginals.

Work package C of the qubit ``[[5,1,2]]`` decoder study. The decoder is meant
for real-time hardware (FPGA-class): at run time it only does table lookups,
XORs and shifts, and its state between rounds is five bits.

Time structure
--------------
Detectors of an ``LRBCircuit`` fall into time slots ``t = 0..T-1`` (one per
check round, plus the final data-readout slot with the ``data5`` terminal), as
in ``mld.lrb_detector_slots``. Every DEM mechanism touches at most two
consecutive slots. Group mechanisms by their *earliest* slot: ``G_u`` holds
the mechanisms whose detectors lie in slot ``u`` and possibly ``u+1``. Its
exact joint distribution is the kernel

    A_u[x, y, l] = P(G_u flips slot-u bits x, slot-(u+1) bits y, observable l),

computed by XOR-convolving the mechanisms (as ``mld.TransferMatrixMLD`` does
with its latest-slot kernels). The syndrome of slot ``u`` is
``s_u = x_u XOR y_u`` with ``y_u`` coming from ``G_{u-1}``.

Decoder (window ``w``)
----------------------
State: a *carry* ``kappa`` (the slot-``t`` bits explained by already committed
mechanisms, at most 4 bits) and a *frame* bit ``f`` (committed logical flip).

* Interior step ``t = 0 .. T-w-1``: with ``sigma = s_t XOR kappa`` and the
  look-ahead ``s_{t+1} .. s_{t+w-1}``, commit

      (kappa', lam) = argmax  A_t[sigma, kappa', lam] * B_{t+1}(kappa')

  where ``B_{t+1}(y) = P(s_{t+1..t+w-1} | G_t puts y on slot t+1)`` sums over
  the later groups in the window, their observable bits, and the unseen slot
  ``t+w`` (unconstrained). Then ``f ^= lam`` and ``kappa = kappa'``.
  (``commit="carry"`` instead takes ``kappa'`` from the lambda-marginal and
  ``lam`` given that ``kappa'``.)
* Final step ``t_f = max(T-w, 0)``: with ``sigma = s_{t_f} XOR kappa`` and the
  remaining syndromes, flip ``f`` if the window posterior of the parity of
  all remaining observable bits is larger for 1 (marginal MAP over everything
  else, exactly as in MLD).

Each step is one table lookup indexed by the ``sigma`` bits followed by the
look-ahead bits (``4w`` bits for interior steps). Tables are computed offline
from the exact DEM probabilities, so they are the exact MLD decision of the
window problem. When ``w >= T`` there is a single final step over the whole
syndrome and the decoder *is* the exact MLD (``final_llr`` then equals
``TransferMatrixMLD.log_likelihood_ratio``; tested). For ``w < T`` the hard
commits and the truncated look-ahead are the approximations being measured.

Tables are per circuit (they depend on the logical Cliffords between rounds
and on the observable's propagation), which suits LRB: the sequence is known
before the run, so tables can be precomputed and loaded.
"""

from __future__ import annotations

import time
from typing import Optional, Sequence

import numpy as np
import stim

from . import mld as M

COMMIT_RULES = ("joint", "carry")


def _convolve(kernel: np.ndarray, vector: int, probability: float) -> None:
    """In-place Z_2 convolution with the two-point distribution {0: 1-p, v: p}."""
    if probability <= 0.0 or vector == 0:
        return
    index = np.arange(kernel.size) ^ vector
    kernel[:] = (1.0 - probability) * kernel + probability * kernel[index]


def earliest_slot_kernels(dem: stim.DetectorErrorModel,
                          detector_slots: Sequence[int]
                          ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Kernels ``A_u[x, y, l]`` of mechanisms grouped by earliest slot.

    Args:
        dem: Detector error model (observable 0 is the logical).
        detector_slots: Slot of each detector (as ``mld.lrb_detector_slots``).

    Returns:
        ``(kernels, slot_detectors)``: ``kernels[u]`` has shape
        ``(2**n_u, 2**n_{u+1}, 2)`` (``n_T = 0``); ``slot_detectors[u]`` lists
        slot ``u``'s detector indices, bit ``k`` of a slot value being detector
        ``slot_detectors[u][k]``.

    Detector-free mechanisms that flip the observable are folded into
    ``kernels[0]`` (they shift every decision equally).
    """
    slots = np.asarray(detector_slots, dtype=np.int64)
    if len(slots) != dem.num_detectors:
        raise ValueError("Need one slot per detector.")
    num_slots = int(slots.max()) + 1 if len(slots) else 1
    slot_detectors = [np.flatnonzero(slots == t) for t in range(num_slots)]
    sizes = [len(d) for d in slot_detectors]
    if max(sizes) > 8:
        raise ValueError("Slots larger than 8 detectors are not supported.")
    position = np.zeros(len(slots), dtype=np.int64)
    for dets in slot_detectors:
        position[dets] = np.arange(len(dets))

    def next_size(u: int) -> int:
        return sizes[u + 1] if u + 1 < num_slots else 0

    flat = []
    for u in range(num_slots):
        kernel = np.zeros(2 ** (sizes[u] + next_size(u) + 1))
        kernel[0] = 1.0
        flat.append(kernel)
    for mech in M.dem_mechanisms(dem):
        if not mech.detectors:
            if mech.flips_observable:
                _convolve(flat[0], 1 << (sizes[0] + next_size(0)),
                          mech.probability)
            continue
        mech_slots = slots[list(mech.detectors)]
        u = int(mech_slots.min())
        if int(mech_slots.max()) > u + 1:
            raise ValueError("Mechanism spans non-adjacent slots.")
        n_u = sizes[u]
        vector = 0
        for det in mech.detectors:
            if slots[det] == u:
                vector ^= 1 << int(position[det])
            else:
                vector ^= 1 << (n_u + int(position[det]))
        if mech.flips_observable:
            vector ^= 1 << (n_u + next_size(u))
        _convolve(flat[u], vector, mech.probability)

    kernels = []
    for u in range(num_slots):
        # flat index = x + 2**n_u * (y + 2**n_next * l)
        kernels.append(flat[u].reshape(2, 2 ** next_size(u), 2 ** sizes[u])
                       .transpose(2, 1, 0).copy())
    return kernels, slot_detectors


def _xor_matrix(n: int) -> np.ndarray:
    values = np.arange(2 ** n)
    return values[:, None] ^ values[None, :]


def _normalize_rows(array: np.ndarray) -> np.ndarray:
    scale = array.reshape(array.shape[0], -1).max(axis=1)
    scale[scale <= 0.0] = 1.0
    return array / scale.reshape((-1,) + (1,) * (array.ndim - 1))


class SlidingWindowLUT:
    """Sliding-window LUT decoder (module docstring).

    Args:
        dem: Detector error model of the circuit.
        detector_slots: Slot of each detector.
        window: Number of slots visible to each decision (``w >= 1``):
            the committed slot plus ``w - 1`` look-ahead slots.
        commit: ``"joint"`` (default) or ``"carry"`` interior commit rule.

    Attributes:
        tables: Interior tables, ``tables[t][index] = kappa' | lam << n_{t+1}``.
        final_table: Final-step decisions (bool) by index.
        final_llr: Final-step ``log P(parity=0) - log P(parity=1)``.
        valid: Per interior table, False where the window probability is 0.
        build_seconds: Precomputation time.
    """

    def __init__(self, dem: stim.DetectorErrorModel,
                 detector_slots: Sequence[int], window: int,
                 commit: str = "joint"):
        if window < 1:
            raise ValueError("window must be >= 1.")
        if commit not in COMMIT_RULES:
            raise ValueError(f"commit must be one of {COMMIT_RULES}.")
        start = time.perf_counter()
        self.window = int(window)
        self.commit = commit
        self.kernels, self.slot_detectors = earliest_slot_kernels(
            dem, detector_slots)
        self.sizes = [len(d) for d in self.slot_detectors]
        self.num_slots = len(self.sizes)
        self.num_interior = max(self.num_slots - self.window, 0)
        self.final_start = self.num_interior
        self.tables: list[np.ndarray] = []
        self.valid: list[np.ndarray] = []
        for t in range(self.num_interior):
            table, valid = self._interior_table(t)
            self.tables.append(table)
            self.valid.append(valid)
        self.final_table, self.final_llr = self._final_table()
        self.build_seconds = time.perf_counter() - start
        self.invalid_hits = 0

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------
    def _next_size(self, u: int) -> int:
        return self.sizes[u + 1] if u + 1 < self.num_slots else 0

    def _interior_table(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        w = self.window
        last = t + w - 1  # last visible slot (< num_slots - 1)
        # Backward message over the look-ahead, observable bits summed out.
        message = np.ones((1, 2 ** self.sizes[last + 1]))
        for u in range(last, t, -1):
            marginal = self.kernels[u].sum(axis=2)  # (x_u, y_{u+1})
            contracted = message @ marginal.T  # (R, x_u)
            n_u = self.sizes[u]
            message = contracted[:, _xor_matrix(n_u)]  # (R, s_u, y_u)
            message = _normalize_rows(message.reshape(-1, 2 ** n_u))
        kernel = self.kernels[t]  # (sigma, kappa, lam)
        rows = message.shape[0]
        scores = kernel[None, :, :, :] * message[:, None, :, None]
        n_t, n_next = self.sizes[t], self._next_size(t)
        scores = scores.reshape(rows * 2 ** n_t, 2 ** n_next, 2)
        if self.commit == "joint":
            flat = scores.reshape(scores.shape[0], -1)
            best = flat.argmax(axis=1)
            kappa, lam = best // 2, best % 2
            valid = flat.max(axis=1) > 0.0
        else:
            carry_scores = scores.sum(axis=2)
            kappa = carry_scores.argmax(axis=1)
            chosen = scores[np.arange(scores.shape[0]), kappa]
            lam = (chosen[:, 1] > chosen[:, 0]).astype(np.int64)
            valid = carry_scores.max(axis=1) > 0.0
        kappa = np.where(valid, kappa, 0)
        lam = np.where(valid, lam, 0)
        table = (kappa | (lam << n_next)).astype(np.uint16)
        return table, valid

    def _final_table(self) -> tuple[np.ndarray, np.ndarray]:
        message = np.zeros((1, 1, 2))
        message[0, 0, 0] = 1.0  # nothing after the last slot, parity 0
        final = None
        for u in range(self.num_slots - 1, self.final_start - 1, -1):
            kernel = self.kernels[u]  # (x_u, y_{u+1}, l1)
            even = message[:, :, 0] @ kernel[:, :, 0].T + \
                message[:, :, 1] @ kernel[:, :, 1].T
            odd = message[:, :, 1] @ kernel[:, :, 0].T + \
                message[:, :, 0] @ kernel[:, :, 1].T
            contracted = np.stack([even, odd], axis=-1)  # (R, x_u, l)
            n_u = self.sizes[u]
            if u == self.final_start:
                final = contracted.reshape(-1, 2)  # row = sigma | r << n_u
            else:
                message = contracted[:, _xor_matrix(n_u), :]  # (R, s, y, l)
                message = _normalize_rows(
                    message.reshape(-1, 2 ** n_u, 2))
        with np.errstate(divide="ignore", invalid="ignore"):
            llr = np.log(final[:, 0]) - np.log(final[:, 1])
        decision = final[:, 1] > final[:, 0]
        return decision, llr

    # ------------------------------------------------------------------
    # Run time
    # ------------------------------------------------------------------
    def slot_values(self, detection_events: np.ndarray) -> list[np.ndarray]:
        """Packed per-slot syndrome values, one ``(shots,)`` array per slot."""
        events = np.asarray(detection_events, dtype=bool)
        values = []
        for dets in self.slot_detectors:
            weights = (1 << np.arange(len(dets))).astype(np.int64)
            values.append(events[:, dets].astype(np.int64) @ weights)
        return values

    def _index(self, values: list[np.ndarray], sigma: np.ndarray, t: int,
               last: int) -> np.ndarray:
        index = sigma.copy()
        shift = self.sizes[t]
        for u in range(t + 1, last + 1):
            index |= values[u] << shift
            shift += self.sizes[u]
        return index

    def predict(self, detection_events: np.ndarray,
                return_llr: bool = False):
        """Predicted observable flip per shot (and the final-step LLR)."""
        values = self.slot_values(detection_events)
        shots = values[0].shape[0]
        carry = np.zeros(shots, dtype=np.int64)
        frame = np.zeros(shots, dtype=bool)
        for t in range(self.num_interior):
            index = self._index(values, values[t] ^ carry, t,
                                t + self.window - 1)
            self.invalid_hits += int((~self.valid[t][index]).sum())
            out = self.tables[t][index].astype(np.int64)
            n_next = self.sizes[t + 1]
            carry = out & ((1 << n_next) - 1)
            frame ^= ((out >> n_next) & 1).astype(bool)
        t = self.final_start
        index = self._index(values, values[t] ^ carry, t, self.num_slots - 1)
        frame ^= self.final_table[index]
        if return_llr:
            return frame, self.final_llr[index]
        return frame

    # ------------------------------------------------------------------
    # Hardware footprint
    # ------------------------------------------------------------------
    def table_entries(self) -> int:
        return sum(t.size for t in self.tables) + self.final_table.size

    def table_bits(self) -> int:
        """Total LUT memory in bits (interior: n_{t+1}+1 bits per entry)."""
        bits = sum(table.size * (self.sizes[t + 1] + 1)
                   for t, table in enumerate(self.tables))
        return bits + self.final_table.size

    def index_bits(self) -> int:
        """Largest table address width in bits."""
        widths = [int(np.log2(t.size)) for t in self.tables]
        widths.append(int(np.log2(self.final_table.size)))
        return max(widths)


def build_lut(lrb_circuit, window: int, commit: str = "joint",
              dem: Optional[stim.DetectorErrorModel] = None
              ) -> SlidingWindowLUT:
    """Convenience constructor from an ``LRBCircuit``."""
    dem = lrb_circuit.circuit.detector_error_model() if dem is None else dem
    return SlidingWindowLUT(dem, M.lrb_detector_slots(lrb_circuit), window,
                            commit=commit)
