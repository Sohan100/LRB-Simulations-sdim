"""Exact forward recursion over a chain DEM.

The hidden state before round ``r`` is ``(h, L)``: the persistent syndrome
``h`` that earlier faults leave for round ``r`` and the accumulated logical
shift ``L``. One slot step is

``alpha_{r+1}(h', L) = sum_{h, L'} alpha_r(h, L')
                       * K_r[L - L', h' - T h, s_r - h]``

where ``K_r`` is the slot kernel and ``s_r`` the observed raw syndrome. An
unobserved round (a postselection policy that ignores it, or a decoder that
does not look at it) uses the kernel summed over ``a``; a round constrained
to a *set* of allowed syndromes (used by exact evaluations of postselection
rules) sums over that set. After the last round the carried syndrome is
summed out, or, for variant circuits that read all five data qutrits at the
end, matched against the noiseless terminal X-syndrome ``t`` through
``t = Pi h + (last-slot terminal shift)``.

Because the terminal logical observable is the only quantity that is ever
read out, this filter is already the exact maximum-likelihood decoder: it is
causal (it only ever looks at past rounds) and the final argmax over ``L``
uses the whole record.
"""

from __future__ import annotations

import numpy as np

from . import zmod3
from .chain_dem import SYNDROME_STATES, SYNDROME_WIDTH, TERMINAL_WIDTH, ChainModel

STATE = SYNDROME_STATES * 3
TERMINAL_STATES = 3**TERMINAL_WIDTH
_SUB = zmod3.subtraction_table(SYNDROME_WIDTH)
_SUB_T = zmod3.subtraction_table(TERMINAL_WIDTH)
_LSHIFT = (np.arange(3)[None, :] - np.arange(3)[:, None]) % 3  # [L', L] -> L - L'


def _carry_index(perm: np.ndarray | None) -> np.ndarray:
    """Return ``B[h, h'] = index(h' - T h)``."""
    transported = np.arange(SYNDROME_STATES) if perm is None else perm
    return _SUB[np.arange(SYNDROME_STATES)[None, :], transported[:, None]]


def _allowed_matrix(allowed: np.ndarray, sub: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """
    ``A[a, j] = allowed[a + offsets[j]]`` for the summed-observation kernels.

    With ``allowed`` a boolean mask over observed values and ``offsets[j]``
    the hidden contribution of row ``j``, ``K @ A`` sums the kernel over every
    shift ``a`` that makes the observed value allowed.
    """
    size = sub.shape[0]
    # index(a + o) = sub[a, (-o)]; build the additive table from sub.
    neg = sub[0, :]  # index(0 - o) = index(-o)
    add = sub[np.arange(size)[:, None], neg[offsets][None, :]]  # [a, j] = index(a + o_j)
    return allowed[add].astype(np.float64)


def _normalize_observation(observed, size: int):
    """Return ``('all' | 'value' | 'set', payload)`` for an observation spec."""
    if observed is None:
        return "all", None
    if isinstance(observed, (int, np.integer)):
        return "value", int(observed)
    mask = np.asarray(observed, dtype=bool)
    if mask.shape != (size,):
        raise ValueError("allowed-set observation must be a boolean mask")
    return "set", mask


def transition_matrix(kernel: np.ndarray, perm, observed) -> np.ndarray:
    """
    Build the ``(243, 243)`` transition matrix of one non-final slot.

    Rows index ``(h, L')`` and columns ``(h', L)`` with ``state = 3*h + L``.

    Args:
        kernel (np.ndarray): ``(3, 81, 81)`` slot kernel ``[l, b, a]``.
        perm: Transfer permutation into the next round.
        observed: ``None`` (round unobserved), an ``int`` syndrome index, or a
            boolean mask over the 81 syndromes (round constrained to a set).

    Returns:
        np.ndarray: Transition matrix.
    """
    carry = _carry_index(perm)  # [h, h']
    kind, payload = _normalize_observation(observed, SYNDROME_STATES)
    hidden = np.arange(SYNDROME_STATES)
    if kind == "all":
        k_ab = kernel.sum(axis=2)  # [l, b]
        block = k_ab[_LSHIFT[None, :, None, :], carry[:, None, :, None]]
    elif kind == "value":
        a_index = _SUB[payload, hidden]  # [h]
        block = kernel[_LSHIFT[None, :, None, :], carry[:, None, :, None], a_index[:, None, None, None]]
    else:
        summed = kernel @ _allowed_matrix(payload, _SUB, hidden)  # [l, b, h]
        block = summed[_LSHIFT[None, :, None, :], carry[:, None, :, None], hidden[:, None, None, None]]
    return np.ascontiguousarray(block).reshape(STATE, STATE)


def final_matrix(kernel: np.ndarray, observed, observed_terminal=None, terminal_perm=None) -> np.ndarray:
    """
    Build the ``(243, 3)`` matrix of the final slot.

    Args:
        kernel (np.ndarray): ``(3, B, 81)`` kernel of the last slot, with
            ``B = 1`` (no terminal detectors) or ``B = 9``.
        observed: Observation of the last round (see ``transition_matrix``).
        observed_terminal: Observation of the terminal X-syndrome (``None``,
            an ``int`` in ``0..8`` or a boolean mask over 9 values). Ignored
            when the kernel has no terminal axis.
        terminal_perm (np.ndarray | None): Map ``h -> Pi h`` (terminal index).

    Returns:
        np.ndarray: Matrix mapping ``(h, L')`` to the final logical ``L``.
    """
    hidden = np.arange(SYNDROME_STATES)
    kind, payload = _normalize_observation(observed, SYNDROME_STATES)
    if kind == "all":
        k_lta = kernel.sum(axis=2, keepdims=True)  # [l, t, 1]
        a_index = np.zeros(SYNDROME_STATES, dtype=np.int64)
    elif kind == "value":
        k_lta = kernel
        a_index = _SUB[payload, hidden]
    else:
        k_lta = kernel @ _allowed_matrix(payload, _SUB, hidden)  # [l, t, h]
        a_index = hidden
    has_terminal = kernel.shape[1] > 1
    t_kind, t_payload = _normalize_observation(observed_terminal, TERMINAL_STATES)
    if not has_terminal or t_kind == "all":
        k_la = k_lta.sum(axis=1)  # [l, a or h]
        block = k_la[_LSHIFT[None, :, :], a_index[:, None, None]]
    else:
        pi_h = terminal_perm[hidden]
        if t_kind == "value":
            t_index = _SUB_T[t_payload, pi_h]  # [h]
            block = k_lta[_LSHIFT[None, :, :], t_index[:, None, None], a_index[:, None, None]]
        else:
            allowed_t = _allowed_matrix(t_payload, _SUB_T, pi_h)  # [t, h]
            # sum_t k_lta[l, t, a_index[h]] * allowed_t[t, h]
            gathered = k_lta[:, :, a_index]  # [l, t, h]
            k_lh = np.einsum("lth,th->lh", gathered, allowed_t)
            block = k_lh[_LSHIFT[None, :, :], hidden[:, None, None]]
    return np.ascontiguousarray(block).reshape(STATE, 3)


def _slot_matrix(chain: ChainModel, slot: int, observed, observed_terminal=None) -> np.ndarray:
    kernel = chain.kernels[slot]
    if slot == chain.num_rounds - 1:
        return final_matrix(kernel, observed, observed_terminal, chain.terminal_perm)
    return transition_matrix(kernel, chain.transfer_perm[slot + 1], observed)


def forward_posteriors(
    chain: ChainModel,
    syndromes: np.ndarray,
    observed_rounds: np.ndarray | None = None,
    terminal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute ``P(L | observed syndromes)`` and the log-evidence for each row.

    Args:
        chain (ChainModel): Chain model of the circuit.
        syndromes (np.ndarray): ``(n, rounds)`` syndrome indices (0..80).
        observed_rounds (np.ndarray | None): Boolean mask over rounds; rounds
            set to ``False`` are marginalized. ``None`` observes all rounds.
        terminal (np.ndarray | None): ``(n,)`` terminal X-syndrome indices
            (0..8) for circuits with terminal detectors; ``None`` ignores it.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(n, 3)`` posteriors and ``(n,)``
        natural-log evidence of the observations.
    """
    syndromes = np.asarray(syndromes, dtype=np.int64)
    n, rounds = syndromes.shape
    if rounds != chain.num_rounds:
        raise ValueError("syndrome history length does not match the chain model")
    if observed_rounds is None:
        observed_rounds = np.ones(rounds, dtype=bool)
    alpha = np.zeros((n, STATE), dtype=np.float64)
    alpha[:, 0] = 1.0
    log_norm = np.zeros(n, dtype=np.float64)
    for slot in range(rounds):
        last = slot == rounds - 1
        new_alpha = np.empty((n, 3 if last else STATE), dtype=np.float64)
        if observed_rounds[slot] or (last and terminal is not None):
            values = syndromes[:, slot] if observed_rounds[slot] else np.full(n, -1)
            t_values = terminal if (last and terminal is not None) else np.full(n, -1)
            keys = (values + 1) * (TERMINAL_STATES + 1) + (t_values + 1)
            for key in np.unique(keys):
                rows = np.flatnonzero(keys == key)
                value = int(values[rows[0]])
                t_value = int(t_values[rows[0]])
                matrix = _slot_matrix(chain, slot, None if value < 0 else value,
                                      None if t_value < 0 else t_value)
                new_alpha[rows] = alpha[rows] @ matrix
        else:
            new_alpha[:] = alpha @ _slot_matrix(chain, slot, None)
        totals = new_alpha.sum(axis=1)
        with np.errstate(divide="ignore"):
            log_norm += np.log(totals)
        safe = np.where(totals > 0, totals, 1.0)
        alpha = new_alpha / safe[:, None]
    return alpha, log_norm


def forward_posteriors_shared(
    chain: ChainModel,
    syndromes: np.ndarray,
    observed_rounds: np.ndarray | None = None,
    terminal: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Same result as ``forward_posteriors`` but shares work across prefixes.

    Rows with identical syndrome prefixes share their forward message, so the
    recursion only runs once per distinct prefix. This is exact and much
    faster when most shots share long trivial stretches.

    Args:
        chain (ChainModel): Chain model of the circuit.
        syndromes (np.ndarray): ``(n, rounds)`` syndrome indices (0..80).
        observed_rounds (np.ndarray | None): Boolean mask over rounds.
        terminal (np.ndarray | None): ``(n,)`` terminal X-syndrome indices.

    Returns:
        tuple[np.ndarray, np.ndarray]: ``(n, 3)`` posteriors and ``(n,)``
        log-evidence.
    """
    syndromes = np.asarray(syndromes, dtype=np.int64)
    n, rounds = syndromes.shape
    if rounds != chain.num_rounds:
        raise ValueError("syndrome history length does not match the chain model")
    if observed_rounds is None:
        observed_rounds = np.ones(rounds, dtype=bool)
    prefix = np.zeros(n, dtype=np.int64)
    alpha = np.zeros((1, STATE), dtype=np.float64)
    alpha[0, 0] = 1.0
    log_norm = np.zeros(1, dtype=np.float64)
    for slot in range(rounds):
        last = slot == rounds - 1
        use_terminal = last and terminal is not None
        if observed_rounds[slot] or use_terminal:
            values = syndromes[:, slot] if observed_rounds[slot] else np.full(n, -1)
            t_values = terminal if use_terminal else np.full(n, -1)
            obs = (values + 1) * (TERMINAL_STATES + 1) + (t_values + 1)
            width = (SYNDROME_STATES + 1) * (TERMINAL_STATES + 1)
            unique_keys, prefix = np.unique(prefix * width + obs, return_inverse=True)
            prefix = prefix.ravel()
            parents, obs_keys = np.divmod(unique_keys, width)
            new_alpha = np.empty((len(unique_keys), 3 if last else STATE))
            for key in np.unique(obs_keys):
                rows = np.flatnonzero(obs_keys == key)
                value, t_value = divmod(int(key), TERMINAL_STATES + 1)
                matrix = _slot_matrix(chain, slot, None if value == 0 else value - 1,
                                      None if t_value == 0 else t_value - 1)
                new_alpha[rows] = alpha[parents[rows]] @ matrix
            log_norm = log_norm[parents]
        else:
            new_alpha = alpha @ _slot_matrix(chain, slot, None)
        totals = new_alpha.sum(axis=1)
        with np.errstate(divide="ignore"):
            log_norm = log_norm + np.log(totals)
        safe = np.where(totals > 0, totals, 1.0)
        alpha = new_alpha / safe[:, None]
    return alpha[prefix], log_norm[prefix]


def constrained_joint(chain: ChainModel, round_observations, terminal_observation=None) -> np.ndarray:
    """
    Exact ``P(L, every round's syndrome in its allowed set)``.

    Args:
        chain (ChainModel): Chain model.
        round_observations: One observation spec per round (``None``, an
            ``int`` or a boolean mask over 81 syndromes).
        terminal_observation: Observation spec for the terminal X-syndrome.

    Returns:
        np.ndarray: ``(3,)`` joint probabilities.
    """
    alpha = np.zeros(STATE)
    alpha[0] = 1.0
    rounds = chain.num_rounds
    for slot in range(rounds):
        alpha = alpha @ _slot_matrix(chain, slot, round_observations[slot],
                                     terminal_observation if slot == rounds - 1 else None)
    return alpha


def detection_joint(chain: ChainModel, used_rounds: np.ndarray, use_terminal: bool = True) -> np.ndarray:
    """
    Exact ``P(L, all used rounds trivial)`` for LRB-D postselection.

    Args:
        chain (ChainModel): Chain model of the circuit.
        used_rounds (np.ndarray): Boolean mask of rounds used for rejection.
        use_terminal (bool): Also require a trivial terminal X-syndrome when
            the circuit has terminal detectors.

    Returns:
        np.ndarray: ``(3,)`` unnormalized joint probabilities; their sum is
        the acceptance probability.
    """
    observations = [0 if used else None for used in used_rounds]
    terminal = 0 if (use_terminal and chain.has_terminal) else None
    return constrained_joint(chain, observations, terminal)
