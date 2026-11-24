"""Decision rules for correction, detection, and hybrid postselection.

Every rule maps one shot's raw syndrome history to an accept flag and a
predicted logical shift ``L_hat``; the corrected logical outcome is the raw
outcome minus ``L_hat`` (a Pauli-frame update of the terminal readout).

Exact rules (no sampling): ``lrbd_exact`` reproduces the repository's LRB-D
postselection policies and ``blip_forgiving_exact`` evaluates detection that
ignores isolated single-round syndrome blips. Sampled rules use the forward
posteriors (``mld_decisions``, soft thresholds) or a hard-decision sliding
window (``window_decode``) that models a real-time decoder with a fixed
lookahead.
"""

from __future__ import annotations

import numpy as np

from . import zmod3
from .chain_dem import SYNDROME_STATES, SYNDROME_WIDTH, ChainModel
from .forward import STATE, _carry_index, _slot_matrix

_SUB = zmod3.subtraction_table(SYNDROME_WIDTH)
_SUB_T = zmod3.subtraction_table(2)
_DIGITS = zmod3.index_to_digits(np.arange(SYNDROME_STATES), SYNDROME_WIDTH)
# Round detectors are ordered by ancilla wire (5, 6, 7, 8) = (S_Z1, S_Z2, S_X1, S_X2).
X_PART_TRIVIAL = np.all(_DIGITS[:, 2:] == 0, axis=1)


def unif_mask(depth: int, interval: int) -> np.ndarray:
    """
    Rounds used by the pipeline's ``unif <interval>`` LRB-D policy.

    Mirrors ``LRBSimulationPipeline.postselect_uniform_interval``.

    Args:
        depth (int): Benchmark depth m (rounds 0..m).
        interval (int): Check interval k >= 1.

    Returns:
        np.ndarray: Boolean mask over the m + 1 rounds.
    """
    mask = np.zeros(depth + 1, dtype=bool)
    mask[depth] = True
    if depth >= interval and interval > 0:
        for j in range(1, depth // interval + 1):
            index = j * interval - 1
            if index < depth:
                mask[index] = True
    return mask


def const_mask(depth: int, num_checks: int) -> np.ndarray:
    """
    Rounds used by the pipeline's ``const <num_checks>`` LRB-D policy.

    Mirrors ``LRBSimulationPipeline.postselect_constant_number`` for
    ``num_checks >= 1`` (``const 0`` uses separate circuits).

    Args:
        depth (int): Benchmark depth m.
        num_checks (int): Number of checks c >= 1.

    Returns:
        np.ndarray: Boolean mask over the m + 1 rounds.
    """
    mask = np.zeros(depth + 1, dtype=bool)
    mask[depth] = True
    if 0 < num_checks <= depth:
        interval = depth // num_checks
        for j in range(1, num_checks):
            mask[j * interval - 1] = True
    return mask


class _SlotMatrices:
    """
    Cache of per-slot transition matrices for one chain model.

    When the circuit has terminal X-stabilizer detectors, every final-slot
    matrix built here requires a trivial terminal syndrome if
    ``use_terminal`` is set (a free, noiseless extra check), and ignores it
    otherwise.
    """

    def __init__(self, chain: ChainModel, use_terminal: bool = True):
        self.chain = chain
        self.terminal = 0 if (use_terminal and chain.has_terminal) else None
        self._cache: dict[tuple, np.ndarray] = {}

    def zero(self, slot: int) -> np.ndarray:
        return self.get(slot, "zero")

    def marg(self, slot: int) -> np.ndarray:
        return self.get(slot, "marg")

    def get(self, slot: int, kind: str) -> np.ndarray:
        key = (slot, kind)
        if key not in self._cache:
            observed = {"zero": 0, "marg": None, "xtriv": X_PART_TRIVIAL}[kind]
            last = slot == self.chain.num_rounds - 1
            self._cache[key] = _slot_matrix(self.chain, slot, observed, self.terminal if last else None)
        return self._cache[key]


def lrbd_exact(chain: ChainModel, used_rounds: np.ndarray, cache: _SlotMatrices | None = None) -> np.ndarray:
    """
    Exact ``P(L, accepted)`` for LRB-D with the given used rounds.

    Args:
        chain (ChainModel): Chain model.
        used_rounds (np.ndarray): Boolean mask of rounds used for rejection.
        cache (_SlotMatrices | None): Optional matrix cache.

    Returns:
        np.ndarray: ``(3,)`` joint probabilities (sum = acceptance).
    """
    cache = cache or _SlotMatrices(chain)
    alpha = np.zeros(STATE)
    alpha[0] = 1.0
    for slot in range(chain.num_rounds):
        matrix = cache.zero(slot) if used_rounds[slot] else cache.marg(slot)
        alpha = alpha @ matrix
    return alpha


def blip_forgiving_exact(
    chain: ChainModel,
    observed_rounds: np.ndarray | None = None,
    cache: _SlotMatrices | None = None,
    final_rule: str = "zero",
) -> np.ndarray:
    """
    Exact ``P(L, accepted)`` for blip-forgiving detection (BFD).

    BFD accepts a shot iff the final round is trivial and no two consecutive
    observed rounds are both non-trivial; it never applies a correction. At
    first order an isolated non-trivial round can only come from a
    measurement-type fault (a persistent data error is seen again in the next
    round), so BFD forgives measurement and ancilla faults while still
    rejecting every single data fault.

    Args:
        chain (ChainModel): Chain model.
        observed_rounds (np.ndarray | None): Rounds the rule may look at
            (unobserved rounds are marginalized and do not break adjacency
            of the observed ones). ``None`` observes all rounds.
        cache (_SlotMatrices | None): Optional matrix cache.
        final_rule (str): ``zero`` requires a trivial final round;
            ``xtriv`` (observable-aware) only requires the final round's
            X-check part to be trivial, because an X-type error left after
            the last check cannot change the logical-X readout.

    Returns:
        np.ndarray: ``(3,)`` joint probabilities (sum = acceptance).
    """
    cache = cache or _SlotMatrices(chain)
    rounds = chain.num_rounds
    if observed_rounds is None:
        observed_rounds = np.ones(rounds, dtype=bool)
    clean = np.zeros(STATE)
    clean[0] = 1.0  # previous observed round trivial
    flagged = np.zeros(STATE)  # previous observed round non-trivial
    for slot in range(rounds):
        zero = cache.zero(slot)
        if not observed_rounds[slot]:
            marg = cache.marg(slot)
            clean, flagged = clean @ marg, flagged @ marg
            continue
        if slot == rounds - 1:
            final = cache.get(slot, final_rule)
            return clean @ final + flagged @ final
        nonzero = cache.marg(slot) - zero
        clean, flagged = clean @ zero + flagged @ zero, clean @ nonzero
    raise AssertionError("unreachable")


def blip_forgiving_accept(
    syndromes: np.ndarray,
    observed_rounds: np.ndarray | None = None,
    terminal: np.ndarray | None = None,
    final_rule: str = "zero",
) -> np.ndarray:
    """
    Per-shot BFD accept flags for sampled syndrome histories.

    Args:
        syndromes (np.ndarray): ``(n, rounds)`` syndrome indices.
        observed_rounds (np.ndarray | None): Rounds the rule may look at.
        terminal (np.ndarray | None): Terminal X-syndrome indices, required
            to be trivial when given.
        final_rule (str): ``zero`` or ``xtriv`` (see ``blip_forgiving_exact``).

    Returns:
        np.ndarray: ``(n,)`` boolean accept flags.
    """
    if observed_rounds is not None:
        syndromes = syndromes[:, observed_rounds]
    nontrivial = syndromes != 0
    consecutive = np.any(nontrivial[:, 1:] & nontrivial[:, :-1], axis=1)
    if final_rule == "zero":
        final_ok = ~nontrivial[:, -1]
    else:
        final_ok = X_PART_TRIVIAL[syndromes[:, -1]]
    accept = final_ok & ~consecutive
    if terminal is not None:
        accept &= terminal == 0
    return accept


def window_decode(
    chain: ChainModel,
    syndromes: np.ndarray,
    lookahead: int,
    terminal: np.ndarray | None = None,
) -> np.ndarray:
    """
    Hard-decision sliding-window decoder with commit size one slot.

    At slot ``t`` the decoder starts from its committed persistent syndrome
    ``h_t``, forms the joint posterior of ``(h_{t+1}, slot-t logical shift)``
    using the observations of rounds ``t .. t + lookahead`` (a backward
    message summarizes the lookahead rounds), commits the argmax, and moves
    on. This models a real-time decoder that must commit each round's
    correction after ``lookahead`` further rounds. With ``lookahead`` covering
    the whole record it is a hard-decision smoother, still not identical to
    the marginal maximum-likelihood rule used by ``forward_posteriors``.

    Args:
        chain (ChainModel): Chain model.
        syndromes (np.ndarray): ``(n, rounds)`` syndrome indices.
        lookahead (int): Number of future rounds used per decision (>= 0).
        terminal (np.ndarray | None): Terminal X-syndrome indices, used in the
            final decision when the circuit has terminal detectors.

    Returns:
        np.ndarray: ``(n,)`` predicted logical shifts.
    """
    syndromes = np.asarray(syndromes, dtype=np.int64)
    n, rounds = syndromes.shape
    kernels = chain.kernels
    # Syndrome-only transition matrices M[s][h, h'] (logical summed out).
    reduced: list[dict[int, np.ndarray]] = [dict() for _ in range(rounds)]

    def reduced_matrix(slot: int, value: int) -> np.ndarray:
        if value not in reduced[slot]:
            if slot == rounds - 1:
                k_a = kernels[slot].sum(axis=(0, 1))  # [a]
                reduced[slot][value] = k_a[_SUB[value, np.arange(SYNDROME_STATES)]][:, None]
            else:
                carry = _carry_index(chain.transfer_perm[slot + 1])
                k_ba = kernels[slot].sum(axis=0)  # [b, a]
                a_index = _SUB[value, np.arange(SYNDROME_STATES)]
                reduced[slot][value] = k_ba[carry, a_index[:, None]]
        return reduced[slot][value]

    committed_h = np.zeros(n, dtype=np.int64)
    logical = np.zeros(n, dtype=np.int64)
    l_index = np.arange(3)
    for slot in range(rounds):
        s_now = syndromes[:, slot]
        a_now = _SUB[s_now, committed_h]
        if slot == rounds - 1:
            if terminal is not None and chain.has_terminal:
                t_index = _SUB_T[terminal, chain.terminal_perm[committed_h]]
                scores = kernels[slot][:, t_index, a_now].T  # (n, 3)
            else:
                scores = kernels[slot].sum(axis=1)[:, a_now].T  # (n, 3)
            logical = (logical + scores.argmax(axis=1)) % 3
            break
        carry = _carry_index(chain.transfer_perm[slot + 1])
        # joint[n, l, h'] = K[l, h' - T h_t, s_t - h_t]
        joint = kernels[slot][l_index[None, :, None], carry[committed_h][:, None, :], a_now[:, None, None]]
        horizon = min(rounds - 1, slot + lookahead)
        beta = np.ones((n, SYNDROME_STATES))
        for future in range(horizon, slot, -1):
            values = syndromes[:, future]
            new_beta = np.empty_like(beta) if future < rounds - 1 else np.empty((n, SYNDROME_STATES))
            for value in np.unique(values):
                rows = np.flatnonzero(values == value)
                matrix = reduced_matrix(future, int(value))
                if future == rounds - 1:
                    new_beta[rows] = matrix[:, 0][None, :].repeat(len(rows), axis=0)
                else:
                    new_beta[rows] = beta[rows] @ matrix.T
            beta = new_beta / np.maximum(new_beta.max(axis=1, keepdims=True), 1e-300)
        scores = joint * beta[:, None, :]
        flat = scores.reshape(n, -1).argmax(axis=1)
        l_best, h_best = np.divmod(flat, SYNDROME_STATES)
        logical = (logical + l_best) % 3
        committed_h = h_best
    return logical


def confidence_histogram(confidence: np.ndarray, residual: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Histogram shots by decoder confidence and residual logical shift.

    Bins are over ``log10(1 - confidence)``; ``edges`` must be increasing.
    Bin 0 collects ``1 - confidence`` below ``10**edges[0]`` (including
    exactly confident shots).

    Args:
        confidence (np.ndarray): ``(n,)`` maximum posterior probability.
        residual (np.ndarray): ``(n,)`` residual logical shift (0 = correct).
        edges (np.ndarray): Bin edges in ``log10(1 - confidence)``.

    Returns:
        np.ndarray: ``(len(edges) + 1, 3)`` integer counts.
    """
    with np.errstate(divide="ignore"):
        value = np.log10(np.clip(1.0 - confidence, 0.0, None))
    bins = np.searchsorted(edges, value, side="right")
    counts = np.zeros((len(edges) + 1, 3), dtype=np.int64)
    np.add.at(counts, (bins, residual.astype(np.int64)), 1)
    return counts
