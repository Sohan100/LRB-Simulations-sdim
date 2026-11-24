"""Exact (shot-noise-free) LRB-D statistics from the time-sliced DEM.

For a postselection policy that requires the detectors of some time slots to
be trivial and ignores the others, the acceptance probability and the joint
probability of acceptance and a logical flip follow from the same transfer
matrix sweep as ``mld.TransferMatrixMLD``. The difference is that each slot is
either conditioned on its all-zero value or marginalized:

    conditioned:  psi_t(b, l) = sum_{c, l'} psi_{t-1}(c, l') K_t(c, b, l^l')
    marginalized: psi_t(b, l) = sum_{l'} [sum_a psi_{t-1}(a, l')]
                                         [sum_c K_t(c, b, l^l')]

A rule that rejects when the *raw* syndrome of a round is non-zero is a rule
on difference detectors only when every earlier round is also used, since a
raw syndrome is the XOR of difference detectors back to the start. Raw
policies that skip rounds therefore cannot be written as conditions on
difference-detector slots. ``exact_raw_policy`` handles them by building a
circuit whose detectors are the raw syndromes of the used rounds. Its slots
are then far from local, so it only works for small depths. The local version
``exact_all_rounds`` covers LRB-D ``unif=1``, with or without the final data
check.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .mld import TransferMatrixMLD


def exact_condition(decoder: TransferMatrixMLD,
                    conditioned_slots: Iterable[int]) -> tuple[float, float]:
    """Return ``(P(accept), P(accept and logical flip))``.

    Args:
        decoder: A ``TransferMatrixMLD`` built from the circuit's DEM; its
            kernels define the model.
        conditioned_slots: Slots whose detectors must all be zero; the
            remaining slots are marginalized.
    """
    conditioned = set(int(t) for t in conditioned_slots)
    psi = decoder.initial.copy()
    for t in range(1, decoder.num_slots):
        n_prev = decoder.slot_sizes[t - 1]
        size = 2 ** n_prev
        matrix = decoder.transfer[t]  # rows (c, l'), cols (b, l)
        if (t - 1) in conditioned:
            # s = 0: phi(c, l') = psi(c, l').
            psi = psi @ matrix
        else:
            # Marginalize slot t-1: sum psi over its bits, sum K over c.
            marg = psi.reshape(2, size).sum(axis=1)  # over a, per l'
            block = matrix.reshape(2, size, -1).sum(axis=1)  # (l', (b, l))
            psi = marg @ block
    n_last = decoder.slot_sizes[-1]
    last = decoder.num_slots - 1
    if last in conditioned:
        p0, p1 = psi[0], psi[2 ** n_last]
    else:
        p0 = psi[:2 ** n_last].sum()
        p1 = psi[2 ** n_last:].sum()
    return float(p0 + p1), float(p1)


def exact_all_rounds(decoder: TransferMatrixMLD, num_round_slots: int,
                     final_check: bool) -> tuple[float, float]:
    """LRB-D with every check round used (``unif=1``).

    Args:
        num_round_slots: Number of check-round slots (``LRBCircuit.num_rounds``).
        final_check: Also require the final data-readout detectors (slot
            ``num_round_slots``) to be trivial.
    """
    slots = list(range(num_round_slots))
    if final_check and decoder.num_slots > num_round_slots:
        slots.append(num_round_slots)
    return exact_condition(decoder, slots)


class RawPolicyEvaluator:
    """Exact acceptance and accepted logical error for raw-syndrome policies.

    The sweep state is ``(r, b, l)``: ``r`` is the raw syndrome of the
    previous check round (4 bits), ``b`` the partially accumulated detector
    bits of the current slot, and ``l`` the logical flip. When a round slot is
    complete, its raw syndrome is ``a XOR M r``, with ``M`` the
    back-propagation map from ``LRBCircuit.back``. A used round is conditioned
    on a zero raw syndrome, and that raw syndrome becomes the next ``r``.
    The final data-readout slot (if present) has a raw X syndrome equal to its
    detectors XOR the back-propagated last-round raw values.

    Args:
        lc: The ``LRBCircuit``.
        decoder: ``TransferMatrixMLD`` built from ``lc``'s DEM; it supplies
            the slot kernels.
    """

    def __init__(self, lc, decoder: TransferMatrixMLD):
        from . import circuits as Q
        from . import code as C

        self.num_round_slots = lc.num_rounds
        self.has_final = decoder.num_slots > lc.num_rounds
        self.sizes = decoder.slot_sizes
        self.num_slots = decoder.num_slots
        # Back maps M_t (rows: this slot's stabilizers, cols: previous round's).
        self.back_maps = []
        for k, label in enumerate(lc.rounds):
            m = np.zeros((4, 4), dtype=np.int64)
            if k > 0:
                for s in range(4):
                    for prev in lc.back[label][s]:
                        m[s, prev] = 1
            self.back_maps.append(m)
        if self.has_final:
            subsets = Q._back_subsets(lc.segments[-1], C.X_STABILIZERS)
            m = np.zeros((2, 4), dtype=np.int64)
            if lc.num_rounds > 0:
                for row, subset in enumerate(subsets):
                    for prev in subset:
                        m[row, prev] = 1
            self.back_maps.append(m)
        # Rebuild each slot kernel K_t(a, b, l) from the decoder's matrices.
        self.kernels = [decoder.initial.copy()]
        for t in range(1, self.num_slots):
            n_prev, n_cur = self.sizes[t - 1], self.sizes[t]
            matrix = decoder.transfer[t]
            kernel = np.zeros((2, 2 ** n_cur, 2 ** n_prev))  # [l, b, c]
            for l_out in range(2):
                block = matrix[0:2 ** n_prev,
                               l_out * 2 ** n_cur:(l_out + 1) * 2 ** n_cur]
                kernel[l_out] = block.T  # K(c, b, l_out) with l_in = 0
            self.kernels.append(kernel)

    @staticmethod
    def _bits_to_int(bits: np.ndarray) -> int:
        return int(sum(int(b) << i for i, b in enumerate(bits)))

    def _raw_map(self, t: int, n_prev_raw: int) -> np.ndarray:
        """table[r, a] = raw syndrome of slot t given previous raw r."""
        m = self.back_maps[t]
        n_cur = m.shape[0]
        table = np.zeros((2 ** n_prev_raw, 2 ** n_cur), dtype=np.int64)
        for r in range(2 ** n_prev_raw):
            rbits = np.array([(r >> i) & 1 for i in range(n_prev_raw)])
            mr = (m @ rbits) % 2 if n_prev_raw else np.zeros(n_cur, int)
            mr_int = self._bits_to_int(mr)
            for a in range(2 ** n_cur):
                table[r, a] = a ^ mr_int
        return table

    def evaluate(self, used_round_slots: Iterable[int], final_check: bool
                 ) -> tuple[float, float]:
        """Return ``(P(accept), P(accept and logical flip))``."""
        used = set(int(k) for k in used_round_slots)
        n0 = self.sizes[0]
        # State after slot-0 mechanisms: psi[r, a, l] with r = 0 (no previous
        # round) stored in a 1-entry r axis.
        psi = self.kernels[0].reshape(2, 2 ** n0)  # [l, a]
        psi = psi.T[None, :, :]  # [r=1, a, l]
        n_r = 0  # bits of r
        for t in range(1, self.num_slots + 1):
            n_prev = self.sizes[t - 1]
            # Complete slot t-1 after group-t mechanisms (if any) are applied.
            if t < self.num_slots:
                n_cur = self.sizes[t]
                kernel = self.kernels[t]  # [l, b, c]
                # psi'[r, a, b, l] = sum_{a', l'} psi[r, a', l'] K(a^a', b, l^l')
                size_a = 2 ** n_prev
                a_idx = np.arange(size_a)
                xor = a_idx[:, None] ^ a_idx[None, :]  # [a, a']
                new = np.zeros((psi.shape[0], size_a, 2 ** n_cur, 2))
                for l_out in range(2):
                    for l_in in range(2):
                        k = kernel[l_out ^ l_in]  # [b, c]
                        # K[a^a', b] -> gather c = a ^ a'
                        kk = k[:, xor]  # [b, a, a']
                        new[:, :, :, l_out] += np.einsum(
                            "rq,baq->rab", psi[:, :, l_in], kk)
            else:
                new = psi[:, :, None, :]  # no further mechanisms
                n_cur = 0
            # Map to the raw syndrome of slot t-1 and condition or marginalize.
            table = self._raw_map(t - 1, n_r)  # [r, a] -> raw
            is_round = (t - 1) < self.num_round_slots
            must_be_zero = ((t - 1) in used) if is_round else final_check
            raw_size = 2 ** self.back_maps[t - 1].shape[0]
            nxt = np.zeros((raw_size, new.shape[2], 2))
            flat_raw = table[:new.shape[0], :new.shape[1]].reshape(-1)
            flat_new = new.reshape(-1, new.shape[2], 2)
            if must_be_zero:
                keep = flat_raw == 0
                nxt[0] = flat_new[keep].sum(axis=0)
            else:
                np.add.at(nxt, flat_raw, flat_new)
            if t == self.num_slots:
                total = nxt.sum(axis=(0, 1))
                return float(total.sum()), float(total[1])
            if is_round:
                psi = nxt  # [raw_{t-1}, b, l]
                n_r = self.back_maps[t - 1].shape[0]
            else:  # the final slot is always the last one
                raise AssertionError("final slot must be last")
        raise AssertionError("unreachable")
