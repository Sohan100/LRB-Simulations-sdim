"""Causal (real-time) decoders with bounded lookahead, and Pauli-frame tools.

Work package D of the qubit ``[[5,1,2]]`` correction study. Nothing here
changes the lead's model: the decoders reuse the exact time-slot kernels of
``mld.TransferMatrixMLD`` and the circuits are rebuilt from the same operation
lists and the same ``circuits.emit_operations`` noise insertion.

Causal decoders
---------------
``mld.TransferMatrixMLD`` is a forward filter over time slots (one slot per
check round, plus the final data readout for ``terminal="data5"``). Its state
after slot ``t`` is the pair ``(b_t, l_t)``:

* ``b_t``: the slot-``t`` detector bits produced by fault mechanisms whose
  latest slot is ``t`` ("completed" faults, e.g. a data error before round t);
* ``l_t``: the cumulative flip of the final observable by all mechanisms whose
  latest slot is ``<= t``.

Mechanisms whose latest slot is ``t+1`` may also flip slot-``t`` detectors
(measurement errors, faults between the Z and X readouts of a round, hook
errors). The leftover ``c_t = s_t XOR b_t`` of the observed slot-``t`` bits
``s_t`` must be produced by them.

``CausalWindowMLD`` must *commit*, for every slot ``t`` in time order, using
only the detection events of slots ``<= t + w`` (lookahead ``w`` rounds):

``commit="state"`` (the default; windowed MLD with commit-and-slide)
    commit the posterior mode ``(b̂_t, l̂_t)`` of the boundary state. This is a
    complete Pauli-frame decision: ``b̂_t`` says which slot-``t`` detection
    events are attributed to completed (data-like) faults and which are left
    to faults that continue into slot ``t+1``; ``l̂_t`` is the committed
    logical frame. The decoder then forgets the past and restarts its filter
    from the committed state. Because the DEM is a Markov chain in this
    state, restarting from ``(b̂_t, l̂_t)`` is exactly conditioning on the
    hypothesis that the committed state is the true one.
``commit="logical"``
    commit only the logical frame ``l̂_t`` and keep the soft distribution over
    ``b_t`` (a decoder that must resolve the logical frame every round, for
    example before a non-Clifford gate, but may keep 16 soft syndrome numbers).

The posterior used for a commit is exact: forward message from the committed
state times a backward message over the lookahead window. If the window ends
before the last slot, the unobserved future is marginalised exactly ("open
boundary": the leftover of the last window slot is weighted by the prior
probability that the next slot's mechanisms produce it). The final prediction
is the committed ``l̂`` of the last slot.

``lookahead=None`` lets every commit see all slots (a non-causal diagnostic
that isolates the effect of committing from the effect of the lookahead).

Pauli-frame tools
-----------------
``build_segmented_lrb`` cuts an LRB circuit after every check round so a
classical controller can act between rounds. ``simulate_feedback`` runs a
feedback rule with ``stim.FlipSimulator`` either with the corrections applied
physically mid-circuit or tracked in a Pauli frame that flips the final
readout; with the same seed both runs see the same noise realisation, so the
logical outputs can be compared shot by shot. ``rule_a_feedback_circuits``
builds the linear lookup rule as Stim classically controlled Paulis
(``CX rec[-k] q``) and as a frame observable, for exact DEM comparisons.
``nonclifford_toy`` and ``leakage_toy`` are exact small models of where the
two would differ.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import stim

from . import circuits as Q
from . import code as C
from . import mld as M

# ---------------------------------------------------------------------------
# Causal windowed MLD
# ---------------------------------------------------------------------------


# Exact posterior ties are common: symmetric fault classes (for example idle
# errors on the two members of an ambiguous pair) give mathematically equal
# entries, which floating point then orders by round-off. Candidates within a
# relative tolerance of the maximum are treated as tied and the tie is broken by
# an explicit rule on the logical bit l_t:
#   "l0"     prefer l_t = 0 (the first-index rule; default)
#   "l1"     prefer l_t = 1
#   "keep"   prefer l_t = l_{t-1} (no new logical increment)
#   "flip"   prefer l_t != l_{t-1}
#   "random" uniform choice (seeded)
# Remaining ties in b_t go to the smallest index.
TIE_RULES = ("l0", "l1", "keep", "flip", "random")
TIE_TOL = 1e-9


def _tie_values(rule: str, prev: np.ndarray, rng: np.random.Generator
                ) -> np.ndarray:
    prev = np.asarray(prev, dtype=np.int64)
    if rule == "l0":
        return np.zeros_like(prev)
    if rule == "l1":
        return np.ones_like(prev)
    if rule == "keep":
        return prev.copy()
    if rule == "flip":
        return 1 - prev
    if rule == "random":
        return rng.integers(0, 2, size=prev.shape)
    raise ValueError(rule)


def choose_state(post: np.ndarray, prev_l: np.ndarray, rule: str, tol: float,
                 rng: np.random.Generator):
    """Mode of ``post`` (shots, 2, B) with explicit, round-off-immune ties.

    Returns ``(l, b, tie, dead)``; ``tie`` marks shots where both values of
    ``l`` have an entry within relative ``tol`` of the maximum.
    """
    shots, _, size = post.shape
    flat = post.reshape(shots, -1)
    top = flat.max(axis=1)
    dead = top <= 0
    cand = (flat >= (top * (1.0 - tol))[:, None]) & (flat > 0)
    cand = cand.reshape(shots, 2, size)
    has = cand.any(axis=2)
    tie = has[:, 0] & has[:, 1]
    l_choice = np.where(has[:, 0], 0, 1).astype(np.int64)
    if tie.any():
        l_choice[tie] = _tie_values(rule, np.asarray(prev_l)[tie], rng)
    b_choice = cand[np.arange(shots), l_choice, :].argmax(axis=1)
    return l_choice, b_choice.astype(np.int64), tie, dead


def choose_logical(pl: np.ndarray, prev_l: np.ndarray, rule: str, tol: float,
                   rng: np.random.Generator):
    """Choose ``l`` from ``pl`` (shots, 2) with the same tie convention."""
    top = pl.max(axis=1)
    dead = top <= 0
    tie = (np.abs(pl[:, 1] - pl[:, 0]) <= tol * top) & ~dead
    l_choice = (pl[:, 1] > pl[:, 0]).astype(np.int64)
    if tie.any():
        l_choice[tie] = _tie_values(rule, np.asarray(prev_l)[tie], rng)
    return l_choice, tie, dead


class CausalWindowMLD:
    """Windowed exact-MLD decoder that commits slot by slot with lookahead.

    Args:
        dem: Detector error model with observable 0 as the logical.
        detector_slots: Slot index of every detector
            (``mld.lrb_detector_slots``).
        lookahead: Number of later slots visible when slot ``t`` is
            committed; ``None`` means all slots (non-causal diagnostic).
        commit: ``"state"`` (commit ``(b_t, l_t)``) or ``"logical"`` (commit
            ``l_t`` only).
        base: Optional prebuilt ``TransferMatrixMLD`` for the same DEM, to
            share the kernel construction between decoders.
        tie_rule: How exact posterior ties in ``l_t`` are broken (see
            ``TIE_RULES``); entries within relative ``tie_tol`` of the maximum
            count as tied, so the decision never depends on round-off.
        seed: Seed of the ``"random"`` tie rule.
    """

    def __init__(self, dem: Optional[stim.DetectorErrorModel],
                 detector_slots: Optional[Sequence[int]],
                 lookahead: Optional[int],
                 commit: str = "state",
                 base: Optional[M.TransferMatrixMLD] = None,
                 tie_rule: str = "l0",
                 tie_tol: float = TIE_TOL,
                 seed: int = 0):
        if commit not in ("state", "logical"):
            raise ValueError("commit must be 'state' or 'logical'.")
        if lookahead is not None and lookahead < 0:
            raise ValueError("lookahead must be >= 0 or None.")
        if tie_rule not in TIE_RULES:
            raise ValueError(f"tie_rule must be one of {TIE_RULES}.")
        self.tie_rule = tie_rule
        self.tie_tol = float(tie_tol)
        self.seed = seed
        self.tie_commits = 0
        self.total_commits = 0
        if base is None:
            base = M.TransferMatrixMLD(dem, detector_slots)
        self.base = base
        self.lookahead = lookahead
        self.commit = commit
        self.num_slots = base.num_slots
        self.slot_sizes = list(base.slot_sizes)
        self.slot_detectors = base.slot_detectors
        n0 = self.slot_sizes[0]
        # K0[l, b] and, for t >= 1, K[t][c, l, b] = K_t(c, b, l) of mld.py.
        self.K0 = base.initial.reshape(2, 2 ** n0).copy()
        self.K: list[Optional[np.ndarray]] = [None]
        self.Kbar: list[Optional[np.ndarray]] = [None]
        self.Mopen: list[Optional[np.ndarray]] = [None]
        for t in range(1, self.num_slots):
            n_prev, n_cur = self.slot_sizes[t - 1], self.slot_sizes[t]
            kernel = base.transfer[t][:2 ** n_prev, :].reshape(
                2 ** n_prev, 2, 2 ** n_cur).copy()
            self.K.append(kernel)
            self.Kbar.append(kernel.sum(axis=1))          # [c, b]
            self.Mopen.append(kernel.sum(axis=(1, 2)))    # [c]
        self.dead_shots = 0

    # -- helpers -----------------------------------------------------------
    def slot_values(self, detection_events: np.ndarray) -> np.ndarray:
        """``(shots, num_slots)`` integer slot syndromes."""
        events = np.asarray(detection_events, dtype=bool)
        out = np.zeros((events.shape[0], self.num_slots), dtype=np.int64)
        for t, dets in enumerate(self.slot_detectors):
            weights = (1 << np.arange(len(dets))).astype(np.int64)
            out[:, t] = events[:, dets].astype(np.int64) @ weights
        return out

    def stream(self, shots: int, record: bool = False,
               keep_posteriors: bool = False) -> "CausalStream":
        """Start a streaming (round-by-round) decoding session."""
        return CausalStream(self, shots, record=record,
                            keep_posteriors=keep_posteriors)

    def decode(self, detection_events: np.ndarray, batch_size: int = 32768,
               return_history: bool = False):
        """Predicted observable flips (bool array).

        With ``return_history`` also returns ``(l_hist, b_hist)``: the
        committed cumulative logical and boundary syndrome of every slot,
        shape ``(shots, num_slots)``.
        """
        values = self.slot_values(detection_events)
        shots = values.shape[0]
        pred = np.zeros(shots, dtype=bool)
        l_hist = np.zeros((shots, self.num_slots), dtype=np.int8)
        b_hist = np.zeros((shots, self.num_slots), dtype=np.int16)
        for start in range(0, shots, batch_size):
            chunk = values[start:start + batch_size]
            st = self.stream(len(chunk), record=return_history)
            for t in range(self.num_slots):
                st.push(chunk[:, t])
            pred[start:start + len(chunk)] = st.prediction
            if return_history:
                l_hist[start:start + len(chunk)] = st.l_hist
                b_hist[start:start + len(chunk)] = st.b_hist
        if return_history:
            return pred, l_hist, b_hist
        return pred

    def filter_llr(self, detection_events: np.ndarray) -> np.ndarray:
        """Offline forward filter with the extracted kernels (no commits).

        Must equal ``TransferMatrixMLD.log_likelihood_ratio``; used as a
        consistency check of the kernel extraction.
        """
        values = self.slot_values(detection_events)
        shots = values.shape[0]
        psi = np.broadcast_to(self.K0, (shots,) + self.K0.shape).copy()
        for t in range(1, self.num_slots):
            psi = _filter_step(psi, values[:, t - 1], self.base.transfer[t],
                               self.slot_sizes[t])
        s = values[:, -1]
        idx = np.arange(shots)
        with np.errstate(divide="ignore"):
            return np.log(psi[idx, 0, s]) - np.log(psi[idx, 1, s])

    def lagged_marginals(self, detection_events: np.ndarray,
                         lookahead: Optional[int]) -> list[np.ndarray]:
        """Uncommitted estimates ``P(l_t, b_t | s_0..s_{t+w})`` per slot.

        The exact fixed-lag smoother: no decision is ever committed, so this
        is the best real-time estimate of the boundary state at slot ``t``
        for a decoder with unlimited memory and lookahead ``w``.
        ``lookahead=None`` gives offline smoothing (all slots). Returns one
        normalised ``(shots, 2, 2**n_t)`` array per slot.
        """
        values = self.slot_values(detection_events)
        shots = values.shape[0]
        cols = [values[:, t] for t in range(self.num_slots)]
        last = self.num_slots - 1
        alphas = [np.broadcast_to(self.K0, (shots,) + self.K0.shape).copy()]
        for t in range(1, self.num_slots):
            alphas.append(_filter_step(alphas[-1], cols[t - 1],
                                       self.base.transfer[t],
                                       self.slot_sizes[t]))
        out = []
        if lookahead is None:
            betas: list[Optional[np.ndarray]] = [None] * self.num_slots
            beta = self.backward(cols, last, last)
            betas[last] = beta
            for tp in range(last - 1, -1, -1):
                g = beta @ self.Kbar[tp + 1].T
                lanes = np.arange(2 ** self.slot_sizes[tp])[None, :]
                beta = _normalise(np.take_along_axis(g, cols[tp][:, None]
                                                     ^ lanes, axis=1))
                betas[tp] = beta
        for t in range(self.num_slots):
            if lookahead is None:
                beta = betas[t]
            else:
                beta = self.backward(cols, t, min(t + lookahead, last))
            post = alphas[t] * beta[:, None, :]
            total = post.reshape(shots, -1).sum(axis=1)
            out.append(post / np.where(total > 0, total, 1.0)[:, None, None])
        return out

    # -- messages ----------------------------------------------------------
    def backward(self, values: Sequence[np.ndarray], t: int, end: int
                 ) -> np.ndarray:
        """Backward message ``beta_t(b)`` over slots ``t..end`` (normalised).

        ``beta_t(b) = P(s_t, ..., s_end | b_t = b)`` with the slots after
        ``end`` unobserved (open boundary) unless ``end`` is the last slot.
        It does not depend on ``l`` because the observable is never seen.
        """
        last = self.num_slots - 1
        n_end = self.slot_sizes[end]
        lanes = np.arange(2 ** n_end)[None, :]
        s_end = np.asarray(values[end])[:, None]
        if end == last:
            beta = (lanes == s_end).astype(float)
        else:
            beta = self.Mopen[end + 1][s_end ^ lanes]
        beta = _normalise(beta)
        for tp in range(end - 1, t - 1, -1):
            g = beta @ self.Kbar[tp + 1].T          # g[c] = sum_b' Kbar[c,b'] beta[b']
            lanes = np.arange(2 ** self.slot_sizes[tp])[None, :]
            beta = np.take_along_axis(g, np.asarray(values[tp])[:, None]
                                      ^ lanes, axis=1)
            beta = _normalise(beta)
        return beta


def _normalise(x: np.ndarray) -> np.ndarray:
    scale = x.reshape(x.shape[0], -1).max(axis=1)
    scale = np.where(scale > 0, scale, 1.0)
    return x / scale.reshape((-1,) + (1,) * (x.ndim - 1))


def _filter_step(psi: np.ndarray, s_prev: np.ndarray, transfer: np.ndarray,
                 n_cur: int) -> np.ndarray:
    """One step of the exact forward filter; ``psi`` has shape (shots, 2, A)."""
    shots, _, a_size = psi.shape
    lanes = np.arange(a_size)[None, :]
    idx = np.asarray(s_prev)[:, None] ^ lanes
    g = np.stack([np.take_along_axis(psi[:, 0, :], idx, axis=1),
                  np.take_along_axis(psi[:, 1, :], idx, axis=1)], axis=1)
    out = (g.reshape(shots, 2 * a_size) @ transfer).reshape(shots, 2,
                                                             2 ** n_cur)
    return _normalise(out)


class CausalStream:
    """Round-by-round session of a ``CausalWindowMLD``.

    Call ``push(slot_values)`` once per slot, in time order. The commit for
    slot ``t`` is made as soon as slot ``t + w`` has been pushed (or the last
    slot, whichever comes first), so the decoder never sees later data.
    """

    def __init__(self, decoder: CausalWindowMLD, shots: int,
                 record: bool = False, keep_posteriors: bool = False):
        self.dec = decoder
        self.shots = shots
        self.values: list[np.ndarray] = []
        self.next = 0
        self.l_hat = np.zeros(shots, dtype=np.int64)
        self.b_hat = np.zeros(shots, dtype=np.int64)
        self.fwd: Optional[np.ndarray] = None
        self._beta_all: Optional[list[np.ndarray]] = None
        self.record = record
        if record:
            self.l_hist = np.zeros((shots, decoder.num_slots), dtype=np.int8)
            self.b_hist = np.zeros((shots, decoder.num_slots), dtype=np.int16)
            self.tie_hist = np.zeros((shots, decoder.num_slots), dtype=bool)
        # Normalised commit posteriors P(l_t, b_t | ...), shape (shots, 2, B).
        self.posteriors: Optional[dict[int, np.ndarray]] = (
            {} if keep_posteriors else None)
        self.dead = 0
        self.ties = 0
        self.rng = np.random.default_rng(decoder.seed)

    @property
    def done(self) -> bool:
        return self.next >= self.dec.num_slots

    @property
    def prediction(self) -> np.ndarray:
        if not self.done:
            raise RuntimeError("Not all slots have been committed yet.")
        return self.l_hat.astype(bool)

    def push(self, slot_values: np.ndarray) -> list[int]:
        """Add the next slot's syndromes; return the slots committed now."""
        if len(self.values) >= self.dec.num_slots:
            raise RuntimeError("All slots were already pushed.")
        self.values.append(np.asarray(slot_values, dtype=np.int64))
        newly = []
        last = self.dec.num_slots - 1
        while self.next <= last:
            t = self.next
            w = self.dec.lookahead
            end = last if w is None else min(t + w, last)
            if len(self.values) - 1 < end:
                break
            self._commit(t, end)
            newly.append(t)
            self.next += 1
        return newly

    def _beta(self, t: int, end: int) -> np.ndarray:
        dec = self.dec
        if dec.lookahead is None:
            # Full lookahead: one backward sweep serves every commit.
            if self._beta_all is None:
                last = dec.num_slots - 1
                betas: list[Optional[np.ndarray]] = [None] * dec.num_slots
                beta = dec.backward(self.values, last, last)
                betas[last] = beta
                for tp in range(last - 1, -1, -1):
                    g = beta @ dec.Kbar[tp + 1].T
                    lanes = np.arange(2 ** dec.slot_sizes[tp])[None, :]
                    beta = _normalise(np.take_along_axis(
                        g, self.values[tp][:, None] ^ lanes, axis=1))
                    betas[tp] = beta
                self._beta_all = betas
            return self._beta_all[t]
        return dec.backward(self.values, t, end)

    def _commit(self, t: int, end: int) -> None:
        dec = self.dec
        shots = self.shots
        idx = np.arange(shots)
        n_t = dec.slot_sizes[t]
        if t == 0:
            alpha = np.broadcast_to(dec.K0, (shots, 2, 2 ** n_t)).copy()
        elif dec.commit == "state":
            c = self.values[t - 1] ^ self.b_hat
            alpha = dec.K[t][c]                      # (shots, 2, 2**n_t)
            flip = self.l_hat == 1
            if flip.any():
                alpha[flip] = alpha[flip][:, ::-1, :]
        else:
            alpha = _filter_step(self.fwd, self.values[t - 1],
                                 dec.base.transfer[t], n_t)
        beta = self._beta(t, end)
        post = alpha * beta[:, None, :]
        if self.posteriors is not None:
            total = post.reshape(shots, -1).sum(axis=1)
            self.posteriors[t] = post / np.where(total > 0, total,
                                                 1.0)[:, None, None]
        prev = self.l_hat
        if dec.commit == "state":
            l_new, b_new, tie, dead = choose_state(
                post, prev, dec.tie_rule, dec.tie_tol, self.rng)
            if dead.any():
                # Inconsistent committed history (never observed in tests):
                # fall back to the forward message alone.
                self.dead += int(dead.sum())
                dec.dead_shots += int(dead.sum())
                l2, b2, _, _ = choose_state(alpha[dead], prev[dead],
                                            dec.tie_rule, dec.tie_tol,
                                            self.rng)
                l_new[dead] = l2
                b_new[dead] = b2
            self.l_hat = l_new
            self.b_hat = b_new
        else:
            pl = post.sum(axis=2)
            l_new, tie, dead = choose_logical(pl, prev, dec.tie_rule,
                                              dec.tie_tol, self.rng)
            if dead.any():
                self.dead += int(dead.sum())
                dec.dead_shots += int(dead.sum())
                l2, _, _ = choose_logical(alpha[dead].sum(axis=2), prev[dead],
                                          dec.tie_rule, dec.tie_tol, self.rng)
                l_new[dead] = l2
            self.l_hat = l_new
            fwd = alpha.copy()
            fwd[idx, 1 - self.l_hat, :] = 0.0
            self.fwd = _normalise(fwd)
            self.b_hat = post[idx, self.l_hat, :].argmax(axis=1)
        n_ties = int(tie.sum())
        self.ties += n_ties
        dec.tie_commits += n_ties
        dec.total_commits += shots
        if self.record:
            self.l_hist[:, t] = self.l_hat
            self.b_hist[:, t] = self.b_hat
            self.tie_hist[:, t] = tie


def true_boundary_states(dem: stim.DetectorErrorModel,
                         detector_slots: Sequence[int],
                         errors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Ground-truth ``(b_t, l_t)`` per slot from sampled DEM error data.

    Args:
        errors: ``(shots, num_error_instructions)`` from
            ``dem.compile_sampler().sample(..., return_errors=True)``.

    Returns:
        ``(b, l)``, each ``(shots, num_slots)``: ``b[:, t]`` is the slot-``t``
        detector pattern of fired mechanisms whose latest slot is ``t`` and
        ``l[:, t]`` the cumulative observable flip of fired mechanisms whose
        latest slot is ``<= t`` (detector-free logical mechanisms count in
        slot 0, as in ``mld.TransferMatrixMLD``).
    """
    slots = np.asarray(detector_slots)
    num_slots = int(slots.max()) + 1
    position = np.zeros(len(slots), dtype=np.int64)
    for u in range(num_slots):
        position[slots == u] = np.arange(int((slots == u).sum()))
    mechs = M.dem_mechanisms(dem)
    if errors.shape[1] != len(mechs):
        raise ValueError("Error columns do not match DEM mechanisms.")
    latest = np.zeros(len(mechs), dtype=np.int64)
    pattern = np.zeros(len(mechs), dtype=np.int64)
    flips = np.zeros(len(mechs), dtype=np.int64)
    for k, mech in enumerate(mechs):
        if mech.detectors:
            u = int(slots[list(mech.detectors)].max())
            latest[k] = u
            for det in mech.detectors:
                if slots[det] == u:
                    pattern[k] ^= 1 << int(position[det])
        flips[k] = int(mech.flips_observable)
    shots = errors.shape[0]
    b = np.zeros((shots, num_slots), dtype=np.int64)
    l_slot = np.zeros((shots, num_slots), dtype=np.int64)
    errs = errors.astype(np.int64)
    for t in range(num_slots):
        ks = np.flatnonzero(latest == t)
        if len(ks) == 0:
            continue
        bits = (pattern[ks][:, None] >> np.arange(8)[None, :]) & 1
        b[:, t] = ((errs[:, ks] @ bits) % 2) @ (1 << np.arange(8))
        l_slot[:, t] = (errs[:, ks] @ flips[ks]) % 2
    l_cum = np.bitwise_xor.accumulate(l_slot, axis=1)
    return b, l_cum


def causal_decoders_for(lc: Q.LRBCircuit, lookaheads: Sequence[Optional[int]],
                        commits: Sequence[str] = ("state",),
                        dem: Optional[stim.DetectorErrorModel] = None,
                        tie_rule: str = "l0"
                        ) -> tuple[M.TransferMatrixMLD,
                                   dict[tuple[str, Optional[int]],
                                        CausalWindowMLD]]:
    """Offline MLD plus causal decoders sharing one kernel construction."""
    if dem is None:
        dem = lc.circuit.detector_error_model()
    base = M.TransferMatrixMLD(dem, M.lrb_detector_slots(lc))
    decoders = {}
    for commit in commits:
        for w in lookaheads:
            decoders[(commit, w)] = CausalWindowMLD(None, None, w, commit,
                                                    base=base,
                                                    tie_rule=tie_rule)
    return base, decoders


def offline_prediction(llr: np.ndarray, tol: float = TIE_TOL) -> np.ndarray:
    """Offline MLD decision with the explicit l0 tie rule.

    ``llr = log P(L=0|s) - log P(L=1|s)``; |llr| <= tol counts as a tie and
    predicts 0. (At the final decision a tie has posterior 1/2 either way, so
    the rule cannot change the expected error, only which tied shots fail.)
    """
    return np.asarray(llr) < -tol


# ---------------------------------------------------------------------------
# Fixed-offset LRB fit (mirrors src/lrb/lrb_plotting.py) and statistics
# ---------------------------------------------------------------------------


def safe_weights(errors: np.ndarray) -> np.ndarray:
    """``LRBPlotter._safe_weights``: ``1/sigma^2`` with a floor for zeros."""
    err = np.asarray(errors, dtype=float)
    err = np.where(np.isnan(err), np.inf, err)
    finite = err[np.isfinite(err)]
    median = float(np.median(finite)) if len(finite) else 1.0
    floor = max(median * 1e-3, 1e-12)
    err = np.where(err <= 0, floor, err)
    return 1.0 / (err * err)


def fit_fixed_offset(depths: Sequence[float], means: Sequence[float],
                     errors: Sequence[float], a_fixed: float = 0.0,
                     y_ceiling: Optional[float] = 1.0,
                     f_min: float = 1e-8, f_max: float = 1.0
                     ) -> Optional[dict[str, float]]:
    """Weighted grid-scan fit of ``y = a_fixed + b f^m`` as in the repo.

    Same algorithm as ``LRBPlotter._fit_decay_parameter_fixed_a`` with the
    default ``LRBPlotFitConfig`` (offset 0, points with mean above 1
    dropped, weights ``1/std^2`` with the std over sequences).
    """
    x = np.asarray(depths, dtype=float)
    y = np.asarray(means, dtype=float)
    e = np.asarray(errors, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(e)
    if y_ceiling is not None:
        ok &= y <= y_ceiling
    x, y, e = x[ok], y[ok], e[ok]
    if len(x) < 2:
        return None
    w = safe_weights(e)
    y0 = y - a_fixed
    low, high = f_min, f_max
    best = None
    for _ in range(5):
        f_grid = np.linspace(low, high, 2001)
        z = np.power(f_grid[None, :], x[:, None])
        num = np.sum(w[:, None] * z * y0[:, None], axis=0)
        den = np.sum(w[:, None] * z * z, axis=0)
        den = np.where(den <= 0, np.nan, den)
        b = num / den
        resid = y0[:, None] - b[None, :] * z
        chi2 = np.sum(w[:, None] * resid * resid, axis=0)
        chi2 = np.where(np.isfinite(chi2), chi2, np.inf)
        i = int(np.argmin(chi2))
        best = {"f": float(f_grid[i]), "a": float(a_fixed), "b": float(b[i]),
                "chi2": float(chi2[i]), "n_points": float(len(x))}
        span = (high - low) / 10.0
        low = max(f_min, best["f"] - span)
        high = min(f_max, best["f"] + span)
    return best


# ---------------------------------------------------------------------------
# Pauli frames on the five data qubits
# ---------------------------------------------------------------------------

_SYN_MATRIX = np.zeros((10, 4), dtype=np.uint8)
for _s, _pauli in enumerate(C.STABILIZER_PAULIS):
    for _q, _ch in enumerate(_pauli):
        if _ch in ("Z", "Y"):
            _SYN_MATRIX[_q, _s] ^= 1          # x-part anticommutes with Z
        if _ch in ("X", "Y"):
            _SYN_MATRIX[5 + _q, _s] ^= 1      # z-part anticommutes with X
# The terminal readout measures X on data 0 and 3: a z-part flips it.
_READOUT_VECTOR = np.zeros(10, dtype=np.uint8)
_READOUT_VECTOR[5 + 0] = 1
_READOUT_VECTOR[5 + 3] = 1


def symplectic_matrix(operations: Sequence[C.Operation]) -> np.ndarray:
    """10x10 GF(2) matrix acting on frame row vectors ``[x0..x4, z0..z4]``."""
    tableau = C.data_tableau(operations)
    mat = np.zeros((10, 10), dtype=np.uint8)
    for q in range(C.NUM_DATA):
        for row, image in ((q, tableau.x_output(q)),
                           (5 + q, tableau.z_output(q))):
            xs, zs = image.to_numpy()
            mat[row, :5] = xs
            mat[row, 5:] = zs
    return mat


def pauli_to_bits(pauli: stim.PauliString) -> np.ndarray:
    xs, zs = pauli.to_numpy()
    return np.concatenate([xs, zs]).astype(np.uint8)


class FrameTracker:
    """Vectorised Pauli frames (one per shot) on the five data qubits."""

    def __init__(self, shots: int):
        self.bits = np.zeros((shots, 10), dtype=np.uint8)

    def propagate(self, matrix: np.ndarray) -> None:
        self.bits = (self.bits.astype(np.int64) @ matrix) % 2
        self.bits = self.bits.astype(np.uint8)

    def multiply(self, delta: np.ndarray) -> None:
        self.bits ^= delta.astype(np.uint8)

    def syndrome(self) -> np.ndarray:
        return ((self.bits.astype(np.int64) @ _SYN_MATRIX) % 2).astype(bool)

    def readout_flip(self) -> np.ndarray:
        return ((self.bits.astype(np.int64) @ _READOUT_VECTOR) % 2
                ).astype(bool)


# ---------------------------------------------------------------------------
# Segmented LRB circuits (a classical controller acts between rounds)
# ---------------------------------------------------------------------------


@dataclass
class SegmentedLRB:
    """An LRB circuit cut after every check round.

    ``segments[k]`` (k < num_rounds) ends with check round ``rounds[k]``;
    ``segments[-1]`` is the ideal terminal readout. Concatenating the
    segments gives ``lc.circuit`` without its DETECTOR/OBSERVABLE lines.
    """

    lc: Q.LRBCircuit
    segments: list[stim.Circuit]
    rounds: list[int]
    pre_round_ops: list[list[C.Operation]]
    pre_round_matrices: list[np.ndarray]
    round_records: list[list[int]]
    readout_records: dict[int, int]
    logical_flip_bits: list[np.ndarray]
    syndrome_maps: list[np.ndarray]

    @property
    def num_rounds(self) -> int:
        return len(self.rounds)

    def concatenated(self) -> stim.Circuit:
        out = stim.Circuit()
        for seg in self.segments:
            out += seg
        return out


def build_segmented_lrb(cliffords: Sequence[str], depth: int, p: float, *,
                        terminal: str = "data5",
                        noise: Optional[Q.SI1000] = Q.SI1000()
                        ) -> SegmentedLRB:
    """Rebuild ``circuits.build_lrb_circuit`` (default arguments) in segments.

    ``logical_flip_bits[k]`` is a data Pauli applied right after round ``k``
    that flips the final ``X_L`` readout and has trivial syndrome (the
    propagated ``Z_L``). ``syndrome_maps[k]`` is the 4x4 GF(2) matrix ``A``
    with ``raw_k = A raw_{k-1}`` for a persistent error (from ``lc.back``).
    """
    lc = Q.build_lrb_circuit(cliffords, depth, p, terminal=terminal,
                             noise=noise)
    forward = list(cliffords[:depth])
    segments: list[stim.Circuit] = []
    pre_ops: list[list[C.Operation]] = []
    round_records: list[list[int]] = []
    offset = 0
    cur = stim.Circuit()
    Q.emit_operations(cur, C.encoding_operations(), p, None, C.NUM_QUBITS)
    pending: list[C.Operation] = []
    for descriptor in forward:
        ops = C.logical_clifford_operations(descriptor)
        Q.emit_operations(cur, ops, p, noise, C.NUM_QUBITS,
                          noisy_measurements=False)
        pending.extend(ops)
        recs = Q.emit_operations(cur, C.check_round_operations(), p, noise,
                                 C.NUM_QUBITS, noisy_measurements=True)
        round_records.append([offset + r for r in recs])
        offset += cur.num_measurements
        segments.append(cur)
        pre_ops.append(pending)
        cur, pending = stim.Circuit(), []
    for descriptor in reversed(forward):
        ops = C.logical_clifford_operations(descriptor, inverse=True)
        Q.emit_operations(cur, ops, p, noise, C.NUM_QUBITS,
                          noisy_measurements=False)
        pending.extend(ops)
    recs = Q.emit_operations(
        cur, C.z_stabilizer_check_operations()
        + C.x_stabilizer_check_operations(), p, noise, C.NUM_QUBITS,
        noisy_measurements=True)
    round_records.append([offset + r for r in recs])
    offset += cur.num_measurements
    segments.append(cur)
    pre_ops.append(pending)
    cur = stim.Circuit()
    readout_qubits = (0, 3) if terminal == "data2" else C.DATA_QUBITS
    recs = Q.emit_operations(
        cur, [("H", (q,)) for q in readout_qubits]
        + [("M", (q,)) for q in readout_qubits], p, None, C.NUM_QUBITS)
    readout = {q: offset + r for q, r in zip(readout_qubits, recs)}
    segments.append(cur)

    rounds = list(range(depth)) + [depth]
    if rounds != lc.rounds:
        raise AssertionError("Round layout differs from build_lrb_circuit.")
    mats = [symplectic_matrix(ops) for ops in pre_ops]
    logical = []
    executed: list[C.Operation] = []
    z_logical = stim.PauliString(C.LOGICAL_Z)
    for ops in pre_ops:
        executed = executed + list(ops)
        logical.append(pauli_to_bits(C.data_tableau(executed)(z_logical)))
    syn_maps = []
    for k, label in enumerate(rounds):
        a = np.zeros((4, 4), dtype=np.uint8)
        if k > 0:
            for s in range(4):
                for prev in lc.back[label][s]:
                    a[s, prev] ^= 1
        syn_maps.append(a)
    return SegmentedLRB(lc=lc, segments=segments, rounds=rounds,
                        pre_round_ops=pre_ops, pre_round_matrices=mats,
                        round_records=round_records, readout_records=readout,
                        logical_flip_bits=logical, syndrome_maps=syn_maps)


def strip_annotations(circuit: stim.Circuit) -> list[tuple]:
    """Canonical list of (gate, args, target group) without annotations."""
    out = []
    for inst in circuit.flattened():
        if inst.name in ("DETECTOR", "OBSERVABLE_INCLUDE", "TICK",
                         "SHIFT_COORDS", "QUBIT_COORDS"):
            continue
        for group in inst.target_groups():
            out.append((inst.name, tuple(inst.gate_args_copy()),
                        tuple(str(t) for t in group)))
    return out


# ---------------------------------------------------------------------------
# Feedback rules (decisions between rounds)
# ---------------------------------------------------------------------------


# Lookup tables: stabilizer index -> (Pauli, data qubit) with that syndrome.
# COVARIANT_TABLE generates the group <X0, Z0, X2, Z3>, which the physical
# logical gates (SWAP(2,3)+H, S/S_DAG/CZ, X, Z) map to itself, so a correction
# from it can be postponed through any logical Clifford and re-derived later
# from the syndrome alone ("covariant tie-breaking").
# NONCOVARIANT_TABLE uses X3 and Z4: logical H maps X3 to Z2, whose lookup is
# Z0, leaving Z0 Z2 = Z_L, so its forward corrections change the outcome.
COVARIANT_TABLE = ((("X", 0), ("X", 2), ("Z", 0), ("Z", 3)))
NONCOVARIANT_TABLE = ((("X", 3), ("X", 2), ("Z", 0), ("Z", 4)))


def lookup_correction(syndrome: np.ndarray,
                      table=COVARIANT_TABLE) -> np.ndarray:
    """Linear lookup correction for a raw syndrome ``(shots, 4)``.

    With the default table: S_Z1 -> X0, S_Z2 -> X2, S_X1 -> Z0, S_X2 -> Z3.
    For syndrome (1, 1) this gives X0 X2 (= X1 S_X1) and Z0 Z3 (= Z1 S_Z1),
    which act like the weight-1 corrections X1, Z1 on every code state.
    Returns frame bits ``(shots, 10)``.
    """
    syndrome = np.asarray(syndrome, dtype=bool)
    delta = np.zeros((syndrome.shape[0], 10), dtype=np.uint8)
    for s, (pauli, q) in enumerate(table):
        column = q if pauli == "X" else 5 + q
        delta[:, column] ^= syndrome[:, s]
    return delta


class FeedbackRule:
    """Interface: decisions from frame-adjusted raw syndromes, round by round.

    ``after_round(k, raw)`` receives the round-``k`` raw syndrome as the
    controller sees it *after* accounting for its own earlier corrections
    (in a physical run the measured value, in a frame run the measured value
    XOR the frame's syndrome) and returns the correction to apply now as
    frame bits ``(shots, 10)``. ``final(readout_flips)`` returns extra
    readout flips decided after the readout (frame only in both runs).
    """

    name = "rule"

    def reset(self, seg: SegmentedLRB, shots: int) -> None:
        self.seg = seg
        self.shots = shots

    def after_round(self, k: int, raw: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def final(self, readout_flips: dict[int, np.ndarray]) -> np.ndarray:
        return np.zeros(self.shots, dtype=bool)


class LookupEveryRound(FeedbackRule):
    """Rule A: memoryless linear lookup on every round's raw syndrome (w=0).

    ``table=NONCOVARIANT_TABLE`` gives rule A' (forward corrections matter).
    """

    def __init__(self, table=COVARIANT_TABLE):
        self.table = table
        self.name = ("A_lookup_every_round" if table == COVARIANT_TABLE
                     else "A'_noncovariant_lookup_every_round")

    def after_round(self, k, raw):
        return lookup_correction(raw, self.table)


class ConfirmedLookup(FeedbackRule):
    """Rule B1: correct only syndrome bits confirmed by two rounds (w=1).

    In forward round ``k >= 1`` the correction is the lookup of
    ``raw_k AND (A_k raw_{k-1})`` (bitwise), i.e. stabilizer flips that are
    present now and were present in the previous round (propagated through
    the logical gates). Single measurement errors are never corrected. The
    terminal round applies the plain lookup. Non-linear in the record.
    """

    name = "B1_confirmed_lookup"

    def reset(self, seg, shots):
        super().reset(seg, shots)
        self.prev = np.zeros((shots, 4), dtype=bool)

    def after_round(self, k, raw):
        raw = np.asarray(raw, dtype=bool)
        if k == self.seg.num_rounds - 1:
            delta = lookup_correction(raw)
        elif k == 0:
            delta = lookup_correction(np.zeros_like(raw))
        else:
            propagated = ((self.prev.astype(np.int64)
                           @ self.seg.syndrome_maps[k].T) % 2).astype(bool)
            delta = lookup_correction(raw & propagated)
        self.prev = raw
        return delta


class CausalMLDLogical(FeedbackRule):
    """Rule B2: causal MLD (lookahead w) decisions applied as logical Paulis.

    The detection events of round ``k`` are rebuilt from the controller's raw
    syndromes and pushed into a ``CausalStream``. Whenever the committed
    logical frame changes, the propagated ``Z_L`` (trivial syndrome, flips
    the final ``X_L``) is applied right after that round. Commits that need
    the final data readout are applied as readout flips.
    """

    def __init__(self, lookahead: int, commit: str = "state"):
        self.lookahead = lookahead
        self.commit = commit
        self.name = f"B2_causal_mld_w{lookahead}_{commit}"

    def reset(self, seg, shots):
        super().reset(seg, shots)
        lc = seg.lc
        if not hasattr(seg, "_dem_cache"):
            seg._dem_cache = lc.circuit.detector_error_model()
        self.decoder = CausalWindowMLD(seg._dem_cache,
                                       M.lrb_detector_slots(lc),
                                       self.lookahead, self.commit)
        self.stream = self.decoder.stream(shots)
        self.prev = np.zeros((shots, 4), dtype=bool)
        self.applied = np.zeros(shots, dtype=np.int64)

    def _push(self, values: np.ndarray) -> None:
        self.stream.push(values)

    def after_round(self, k, raw):
        raw = np.asarray(raw, dtype=bool)
        if k == 0:
            det = raw.copy()
        else:
            det = raw ^ ((self.prev.astype(np.int64)
                          @ self.seg.syndrome_maps[k].T) % 2).astype(bool)
        self.prev = raw
        self._push(det.astype(np.int64) @ (1 << np.arange(4)))
        change = (self.stream.l_hat ^ self.applied).astype(bool)
        if self.stream.next == 0:
            change[:] = False
        self.applied ^= change.astype(np.int64)
        delta = np.zeros((self.shots, 10), dtype=np.uint8)
        delta[change] = self.seg.logical_flip_bits[k]
        return delta

    def final(self, readout_flips):
        seg = self.seg
        if seg.lc.terminal == "data5":
            dets = []
            for s in C.X_STABILIZERS:
                support = [q for q, ch in enumerate(C.STABILIZER_PAULIS[s])
                           if ch != "_"]
                value = np.zeros(self.shots, dtype=bool)
                for q in support:
                    value ^= readout_flips[q]
                value ^= self.prev[:, s]      # terminal round, identity in between
                dets.append(value)
            self._push(np.stack(dets, axis=1).astype(np.int64)
                       @ (1 << np.arange(2)))
        if not self.stream.done:
            raise AssertionError("Causal stream did not finish.")
        return (self.stream.l_hat ^ self.applied).astype(bool)


# ---------------------------------------------------------------------------
# Physical-versus-frame simulation with stim.FlipSimulator
# ---------------------------------------------------------------------------


@dataclass
class FeedbackRun:
    """Per-shot results of one feedback simulation."""

    logical_flip: np.ndarray            # final logical output flip per shot
    corrections: list[np.ndarray]       # correction frame bits per round
    raw_seen: list[np.ndarray]          # raw syndromes as seen by the rule


def simulate_feedback(seg: SegmentedLRB, rule: FeedbackRule, shots: int,
                      seed: int, mode: str,
                      correction_noise: float = 0.0,
                      correction_noise_seed: int = 0) -> FeedbackRun:
    """Run ``rule`` with corrections applied physically or in a Pauli frame.

    Args:
        mode: ``"physical"`` applies each correction Pauli to the data qubits
            mid-circuit (FlipSimulator Pauli injection, i.e. an explicit
            conditional gate); ``"frame"`` only records it and flips the
            final readout.
        correction_noise: physical mode only: after each applied correction
            Pauli, the corrected qubit suffers a uniformly random non-identity
            Pauli with this probability (noisy conditional gate).
        seed: FlipSimulator seed. Masked Pauli injections do not consume
            random numbers, so both modes with the same seed see the same
            circuit-noise realisation.
    """
    if mode not in ("physical", "frame"):
        raise ValueError("mode must be 'physical' or 'frame'.")
    sim = stim.FlipSimulator(batch_size=shots, num_qubits=C.NUM_QUBITS,
                             seed=seed)
    frame = FrameTracker(shots)
    noise_rng = np.random.default_rng(correction_noise_seed)
    rule.reset(seg, shots)
    corrections, raw_seen = [], []
    for k in range(seg.num_rounds):
        sim.do(seg.segments[k])
        flips = sim.get_measurement_flips()
        raw = flips[seg.round_records[k], :].T.copy()
        frame.propagate(seg.pre_round_matrices[k])
        if mode == "frame":
            raw ^= frame.syndrome()
        delta = rule.after_round(k, raw)
        raw_seen.append(raw)
        corrections.append(delta)
        x_mask = np.zeros((C.NUM_QUBITS, shots), dtype=bool)
        z_mask = np.zeros((C.NUM_QUBITS, shots), dtype=bool)
        if mode == "physical":
            x_mask[:C.NUM_DATA] = delta[:, :5].T.astype(bool)
            z_mask[:C.NUM_DATA] = delta[:, 5:].T.astype(bool)
        else:
            frame.multiply(delta)
        sim.broadcast_pauli_errors(pauli="X", mask=x_mask)
        sim.broadcast_pauli_errors(pauli="Z", mask=z_mask)
        if mode == "physical" and correction_noise > 0:
            touched = (x_mask | z_mask)
            hit = touched & (noise_rng.random(touched.shape)
                             < correction_noise)
            which = noise_rng.integers(1, 4, size=touched.shape)
            for pauli, code_ in (("X", 1), ("Y", 2), ("Z", 3)):
                sim.broadcast_pauli_errors(pauli=pauli,
                                           mask=hit & (which == code_))
    sim.do(seg.segments[-1])
    flips = sim.get_measurement_flips()
    readout = {q: flips[r, :].copy() for q, r in seg.readout_records.items()}
    out = readout[0] ^ readout[3]
    if mode == "frame":
        out ^= frame.readout_flip()
    out ^= rule.final(readout)
    return FeedbackRun(logical_flip=out, corrections=corrections,
                       raw_seen=raw_seen)


# ---------------------------------------------------------------------------
# Rule A as Stim classically controlled Paulis (exact DEM comparison)
# ---------------------------------------------------------------------------


def rule_a_frame_coefficients(seg: SegmentedLRB,
                              table=COVARIANT_TABLE) -> np.ndarray:
    """GF(2) coefficients of the frame flip of rule A on each ancilla record.

    Rule A is linear, so the frame-tracked readout flip is a parity of
    measurement records. Running the frame logic on unit records gives it.
    """
    n_rec = 4 * seg.num_rounds
    shots = n_rec
    frame = FrameTracker(shots)
    for k in range(seg.num_rounds):
        frame.propagate(seg.pre_round_matrices[k])
        raw = np.zeros((shots, 4), dtype=bool)
        for s in range(4):
            raw[4 * k + s, s] = True
        adjusted = raw ^ frame.syndrome()
        frame.multiply(lookup_correction(adjusted, table))
    return frame.readout_flip()


def rule_a_feedback_circuits(seg: SegmentedLRB,
                             correction_noise: float = 0.0,
                             table=COVARIANT_TABLE
                             ) -> tuple[stim.Circuit, stim.Circuit]:
    """Rule A as (physical feedback circuit, frame circuit).

    Physical: after every check round, one classically controlled Pauli per
    stabilizer record (default ``CX rec[-4] 0``, ``CX rec[-3] 2``,
    ``CZ rec[-2] 0``, ``CZ rec[-1] 3``; records M5, M6, M7, M8 of that
    round), optionally followed by ``DEPOLARIZE1(correction_noise)`` on the
    corrected data qubits every round (an always-played conditional gate).
    Observable 0 is ``x0 + x3``.

    Frame: the same circuit without feedback; observable 0 is
    ``x0 + x3`` plus the frame's parity of ancilla records.
    """
    physical = stim.Circuit()
    frame = stim.Circuit()
    touched = sorted({q for _, q in table})
    for k in range(seg.num_rounds):
        physical += seg.segments[k]
        frame += seg.segments[k]
        for s, (pauli, q) in enumerate(table):
            physical.append("CX" if pauli == "X" else "CZ",
                            [stim.target_rec(s - 4), q])
        if correction_noise > 0:
            physical.append("DEPOLARIZE1", touched, correction_noise)
    physical += seg.segments[-1]
    frame += seg.segments[-1]
    total = frame.num_measurements
    r0, r3 = seg.readout_records[0], seg.readout_records[3]
    physical.append("OBSERVABLE_INCLUDE",
                    [stim.target_rec(r0 - total), stim.target_rec(r3 - total)],
                    0)
    coeffs = rule_a_frame_coefficients(seg, table)
    records = [r0, r3] + [seg.round_records[i // 4][i % 4]
                          for i in np.flatnonzero(coeffs)]
    frame.append("OBSERVABLE_INCLUDE",
                 [stim.target_rec(r - total) for r in records], 0)
    return physical, frame


def observable_flip_probability(circuit: stim.Circuit) -> float:
    """Exact P(observable 0 flips) from the circuit's DEM."""
    dem = circuit.detector_error_model()
    prod = 1.0
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        if any(t.is_logical_observable_id() and t.val == 0
               for t in inst.targets_copy()):
            prod *= 1.0 - 2.0 * inst.args_copy()[0]
    return 0.5 * (1.0 - prod)


# ---------------------------------------------------------------------------
# Where physical correction and frame tracking differ (exact toy models)
# ---------------------------------------------------------------------------


def _encoded_plus_state() -> np.ndarray:
    """Double-precision |+_L> from the encoder (qubit 0 = lowest index bit)."""
    psi = np.zeros(2 ** C.NUM_DATA, dtype=complex)
    psi[0] = 1.0
    index = np.arange(2 ** C.NUM_DATA)
    for gate, targets in C.encoding_operations():
        if gate == "H":
            q = targets[0]
            bit = (index >> q) & 1
            partner = psi[index ^ (1 << q)]
            psi = (np.where(bit == 0, psi + partner, partner - psi)
                   / np.sqrt(2.0))
        elif gate == "CX":
            control, target = targets
            flip = ((index >> control) & 1).astype(bool)
            psi = np.where(flip, psi[index ^ (1 << target)], psi)
        else:
            raise ValueError(f"Unexpected encoder gate {gate}.")
    return psi


def _pauli_matrix(pauli: str) -> np.ndarray:
    """Dense 32x32 matrix of a five-qubit Pauli string (little endian)."""
    single = {"_": np.eye(2), "I": np.eye(2),
              "X": np.array([[0, 1], [1, 0]], dtype=complex),
              "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
              "Z": np.diag([1.0, -1.0]).astype(complex)}
    mat = np.array([[1.0 + 0j]])
    for ch in reversed(pauli):         # qubit 4 is the most significant bit
        mat = np.kron(mat, single[ch])
    return mat


def nonclifford_toy(eps: float) -> dict[str, float]:
    """Exact model: an X0 error, then a logical T, then a Y_L readout.

    ``|+_L>`` suffers ``X0`` with probability ``eps``; an ideal syndrome
    measurement flags it with certainty and the decision is "correct X0".
    Then the logical T gate ``T_L = exp(-i pi/8 Z_L)`` (``Z_L = Z0 Z2``) and
    an ideal ``Y_L = Y0 X3 Z2`` readout follow. Returns ``P(Y_L = +1)`` for:

    * ``physical``: X0 applied before T_L;
    * ``frame_naive``: X0 only tracked in the frame and used to flip the
      readout (correct for Clifford gates, wrong across T);
    * ``frame_fixup``: frame plus the Clifford fix-up ``S_L^dagger`` after
      T_L when the frame anticommutes with ``Z_L`` (needs the decision at the
      T gate).
    """
    psi = _encoded_plus_state()
    zl = _pauli_matrix("Z_Z__")
    yl = _pauli_matrix("Y_ZX_")          # i X_L Z_L = Y0 Z2 X3
    x0 = _pauli_matrix("X____")
    eye = np.eye(32)
    t_l = np.cos(np.pi / 8) * eye - 1j * np.sin(np.pi / 8) * zl
    s_dag = np.cos(np.pi / 4) * eye + 1j * np.sin(np.pi / 4) * zl

    def p_plus(state):
        return float(np.real(np.vdot(state, (eye + yl) @ state)) / 2)

    def anticommutes(a, b):
        return np.allclose(a @ b, -b @ a)

    clean = p_plus(t_l @ psi)
    erred = x0 @ psi
    physical = p_plus(t_l @ (x0 @ erred))
    raw = p_plus(t_l @ erred)
    flip = anticommutes(x0, yl)
    naive = 1 - raw if flip else raw
    fixed = s_dag @ (t_l @ erred) if anticommutes(x0, zl) else t_l @ erred
    fix = p_plus(fixed)
    fix = 1 - fix if flip else fix
    return {
        "eps": eps,
        "no_error": clean,
        "physical": (1 - eps) * clean + eps * physical,
        "frame_naive": (1 - eps) * clean + eps * naive,
        "frame_fixup": (1 - eps) * clean + eps * fix,
    }


def leakage_toy(eps: float, leak: float, seep_to: int,
                basis: str = "Z") -> dict[str, float]:
    """Exact single-transmon model of a correction meeting leakage.

    One data qubit (levels 0, 1, 2) should end in ``|0>`` (basis ``"Z"``) or
    ``|+>`` (basis ``"X"``). An error (X for Z basis, Z for X basis) occurs
    with probability ``eps`` and is flagged perfectly; the decision is to
    correct it. Before the correction the qubit leaks with probability
    ``leak``: for basis ``"Z"`` incoherently (``|1> -> |2>``), for basis
    ``"X"`` coherently (``|1> <-> |2>`` swapped). A physical correction acts
    on levels {0, 1} only (identity on ``|2>``). Before readout the qubit
    returns to the qubit subspace: ``|2> -> |seep_to>`` (incoherent) for
    basis ``"Z"``, the reverse swap for basis ``"X"``. Returns the readout
    error probability for physical and frame correction.
    """
    ket = np.eye(3)
    x_q = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]], dtype=complex)
    z_q = np.diag([1, -1, 1]).astype(complex)
    swap12 = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=complex)
    plus = (ket[0] + ket[1]) / np.sqrt(2)
    minus = (ket[0] - ket[1]) / np.sqrt(2)
    if basis == "Z":
        ideal = np.outer(ket[0], ket[0]).astype(complex)
        err = x_q
    elif basis == "X":
        ideal = np.outer(plus, plus).astype(complex)
        err = z_q
    else:
        raise ValueError("basis must be 'Z' or 'X'.")

    def apply(rho, kraus):
        return sum(k @ rho @ k.conj().T for k in kraus)

    # Incoherent leakage |1> -> |2> (trace preserving) and seepage |2> -> s.
    leak_kraus = [np.outer(ket[2], ket[1]),
                  np.outer(ket[0], ket[0]) + np.outer(ket[2], ket[2])]
    seep_kraus = [np.outer(ket[seep_to], ket[2]),
                  np.outer(ket[0], ket[0]) + np.outer(ket[1], ket[1])]

    def readout_error(rho, frame_flip):
        if basis == "Z":
            wrong = float(np.real(rho[1, 1] + rho[2, 2]))   # outcome 1
        else:
            wrong = float(np.real(minus.conj() @ rho @ minus))  # outcome -
        return 1.0 - wrong if frame_flip else wrong

    out = {"eps": eps, "leak": leak, "seep_to": seep_to, "basis": basis}
    for mode in ("physical", "frame"):
        total = 0.0
        for error, p_err in ((False, 1 - eps), (True, eps)):
            rho = err @ ideal @ err.conj().T if error else ideal
            for leaked, p_leak in ((False, 1 - leak), (True, leak)):
                r = rho
                if leaked:
                    r = (apply(r, leak_kraus) if basis == "Z"
                         else swap12 @ r @ swap12)
                if error and mode == "physical":
                    r = err @ r @ err.conj().T     # identity on |2>
                if leaked:
                    r = (apply(r, seep_kraus) if basis == "Z"
                         else swap12 @ r @ swap12)
                flip = error and mode == "frame"
                total += p_err * p_leak * readout_error(r, flip)
        out[mode] = total
    return out
