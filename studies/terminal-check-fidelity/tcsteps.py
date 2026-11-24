"""Exact per-step factors of terminal-check circuits and what follows from them.

For a fixed sequence, each character of the terminal outcome distribution is
a product over the noisy Cliffords of one factor per Clifford (tclib). The
factor of a Clifford depends only on that Clifford and on the "frame" in
which it acts, i.e. the symplectic class F of the product of the Cliffords
before it (6 classes for the qubit, 24 for the qutrit), because the terminal
X-type operators are carried back to that Clifford through F.

``StepModel`` tabulates these factors exactly from sdim's DEM for

* a mirror pair   (C then its token-wise inverse)  in frame F,
* a single forward Clifford C                      in frame F,
* the compiled recovery Clifford R (acts last, so it sees the X frame),
* the noisy encoder and the noisy terminal readout (SPAM).

From them it gives, with no sampling and for any p:

* the exact sequence-averaged characters at every depth (transfer matrix
  over frames), for the mirror and the compiled protocol;
* ground-truth logical infidelities of one Clifford, one pair, or n of them
  between a perfect preparation and a perfect check;
* exact per-sequence distributions for random sequences, from which
  synthetic finite-shot experiments are drawn (multinomial shots).

Checked in ``02b_exact_checks.py``: the step factors reproduce the directly
built depth-m circuits of the production sequences to 1e-15.
"""
from __future__ import annotations

import numpy as np

import tclib as T
import tcexact as E


def _paulis(d):
    tok = T._token_unitaries(d)
    return [np.linalg.matrix_power(tok["X"], a) @ np.linalg.matrix_power(tok["Z"], b)
            for a in range(d) for b in range(d)]


class StepModel:
    def __init__(self, code: str, noise: str, workers: int = 8, crosstalk=()):
        self.code, self.noise = code, noise
        _, cd, rec = T.setup(code, noise)
        self.d = d = cd.dimension
        self.K = d ** 3
        self.cat = list(cd.clifford_strings)
        self.rec = rec
        paulis = _paulis(d)

        def frame_key(u):
            return min(T._phase_key(p @ u) for p in paulis)

        keys: dict = {}
        self.frame_of_word = np.array([keys.setdefault(frame_key(rec.u[w]), len(keys)) for w in self.cat])
        self.nf = len(keys)
        self.rep = [self.cat[int(np.argmax(self.frame_of_word == f))] for f in range(self.nf)]
        self.f0 = keys[frame_key(np.eye(d))]
        nc = len(self.cat)
        self.compose = np.zeros((self.nf, nc), dtype=np.int64)
        for f in range(self.nf):
            for ci, w in enumerate(self.cat):
                self.compose[f, ci] = keys[frame_key(rec.u[w] @ rec.u[self.rep[f]])]
        # index of the catalog recovery word of each single word (for the compiled protocol)
        self.inv_frame = np.array([self.frame_of_word[self.cat.index(rec.word([self.rep[f]]))]
                                   for f in range(self.nf)])

        mirror = [[(self.rep[f], "fwd_ideal"), (w, "fwd"), (w, "inv"), (self.rep[f], "inv_ideal")]
                  for f in range(self.nf) for w in self.cat]
        forward = [[(self.rep[f], "fwd_ideal"), (w, "fwd"), (rec.word([self.rep[f], w]), "fwd_ideal")]
                   for f in range(self.nf) for w in self.cat]
        recovery = [[(rec.word([w]), "fwd_ideal"), (w, "fwd")] for w in self.cat]
        kw = dict(workers=workers, crosstalk=crosstalk)
        self.t_mirror = E.segments_table(code, noise, mirror, name="step_mirror", **kw)
        self.t_forward = E.segments_table(code, noise, forward, name="step_forward", **kw)
        self.t_recovery = E.segments_table(code, noise, recovery, name="step_recovery", **kw)
        self.t_prep = E.segments_table(code, noise, [[]], noisy_prep=True, name="spam_prep")
        self.t_meas = E.segments_table(code, noise, [[]], noisy_meas=True, name="spam_meas")

    # ------------------------------------------------------------- factors
    @staticmethod
    def _override(p, xt):
        return {("N2", T.CROSSTALK_TAG): (xt / p if p > 0 else 0.0)}

    def factors(self, p: float, kind: str, xt: float = 0.0) -> np.ndarray:
        """c[f, C, k] for kind 'mirror' or 'forward'; c[C, k] for 'recovery'.

        ``xt`` is the absolute probability of the injected correlated noise
        (only if the model was built with ``crosstalk`` pairs).
        """
        ov = self._override(p, xt)
        if kind == "mirror":
            return self.t_mirror.chars(p, ov).reshape(self.nf, len(self.cat), self.K)
        if kind == "forward":
            return self.t_forward.chars(p, ov).reshape(self.nf, len(self.cat), self.K)
        if kind == "recovery":
            return self.t_recovery.chars(p, ov)
        raise ValueError(kind)

    def spam(self, p: float, prep: bool = True, meas: bool = True, meas_scale: float = 1.0) -> np.ndarray:
        """Character factor B_k of the generator's noisy encoder and/or noisy readout.

        ``meas_scale`` multiplies the readout-noise probabilities only.
        """
        b = np.ones(self.K)
        if prep:
            b = b * self.t_prep.chars(p)[0]
        if meas:
            b = b * self.t_meas.chars(p * meas_scale)[0]
        return b

    def transfer(self, p: float, kind: str, xt: float = 0.0) -> np.ndarray:
        """T[k, f_next, f] = mean over the catalog of c[f, C, k] 1[frame(C f) = f_next]."""
        c = self.factors(p, kind, xt)
        nc = len(self.cat)
        tm = np.zeros((self.K, self.nf, self.nf))
        for f in range(self.nf):
            for fn in range(self.nf):
                sel = self.compose[f] == fn
                if sel.any():
                    tm[:, fn, f] = c[f, sel].sum(0) / nc
        return tm

    def _recovery_by_frame(self, p, xt=0.0):
        """Mean recovery-Clifford factor given the frame of the forward product."""
        cr = self.factors(p, "recovery", xt)
        return np.array([cr[self.frame_of_word == self.inv_frame[f]].mean(0) for f in range(self.nf)])

    # --------------------------------------------------- exact expectations
    def exact_chars(self, p: float, depths, protocol: str = "mirror", spam=None, xt: float = 0.0) -> np.ndarray:
        """Sequence-averaged characters chi[depth, k] for uniformly random sequences.

        protocol 'mirror': depth m = m pairs. 'compiled': depth m = m forward
        Cliffords + one recovery Clifford (no gate at m = 0, as in production).
        """
        depths = list(depths)
        tm = self.transfer(p, "mirror" if protocol == "mirror" else "forward", xt)
        v = np.zeros((self.K, self.nf))
        v[:, self.f0] = 1.0
        crf = self._recovery_by_frame(p, xt) if protocol == "compiled" else None
        out = {}
        for m in range(max(depths) + 1):
            if m in depths:
                out[m] = v.sum(1) if (protocol == "mirror" or m == 0) else (v * crf.T).sum(1)
            v = np.einsum("kab,kb->ka", tm, v)
        chi = np.array([out[m] for m in depths])
        return chi if spam is None else chi * spam

    def exact_curve(self, p: float, depths, protocol: str = "mirror", spam=None, xt: float = 0.0) -> dict:
        """Exact pooled acceptance, accepted polarization and infidelity per depth."""
        chi = self.exact_chars(p, depths, protocol, spam, xt)
        P = np.clip(T.dist_from_chars(chi, self.d), 0.0, None)
        A, J, lam = T.accept_and_polarization(P, self.d)
        return {"chi": chi, "P": P, "A": A, "lam": lam, "eps": T.infidelity_from_polarization(lam, self.d)}

    def _eps_of(self, chi):
        _, _, lam = T.accept_and_polarization(T.dist_from_chars(chi, self.d), self.d)
        return float(T.infidelity_from_polarization(lam, self.d))

    def ground_truth(self, p: float, xt: float = 0.0) -> dict:
        """Logical infidelity between a perfect preparation and a perfect check.

        pair_X / pair_stat: one mirror pair in the X frame (= depth 1 of the
        protocol with ideal SPAM) / in a uniformly random frame.
        single_X / single_stat: one Clifford followed by a noiseless recovery.
        Values are (d-1)/d (1 - J/A), pooled over the whole catalog;
        ``*_discard`` is the rejected fraction 1 - A of the same unit.
        """
        out = {}
        for kind, name in (("mirror", "pair"), ("forward", "single")):
            c = self.factors(p, kind, xt)
            for tag, chi in (("X", c[self.f0].mean(0)), ("stat", c.mean((0, 1)))):
                out[f"{name}_{tag}"] = self._eps_of(chi)
                A, _, _ = T.accept_and_polarization(T.dist_from_chars(chi, self.d), self.d)
                out[f"{name}_{tag}_discard"] = float(1 - A)
        return out

    def asymptotic_rates(self, p: float, kind: str = "mirror", xt: float = 0.0) -> np.ndarray:
        """gamma_k = -ln(dominant eigenvalue of the frame transfer matrix of character k).

        These are the exact decay rates per step of the sequence-averaged
        characters at large depth (k = 0 gives 0).
        """
        tm = self.transfer(p, kind, xt)
        return np.array([-np.log(np.max(np.abs(np.linalg.eigvals(tm[k])))) for k in range(self.K)])

    def segment_truth(self, p: float, n_max: int, kind: str = "forward", xt: float = 0.0) -> np.ndarray:
        """Exact infidelity of n i.i.d. steps (n = 1..n_max) between a perfect
        preparation and a perfect check, in a uniformly random frame."""
        tm = self.transfer(p, kind, xt)
        v = np.full((self.K, self.nf), 1.0 / self.nf)
        out = []
        for _ in range(n_max):
            v = np.einsum("kab,kb->ka", tm, v)
            out.append(self._eps_of(v.sum(1)))
        return np.array(out)

    # ----------------------------------------------------------- sampling
    def draw(self, n_seq: int, length: int, rng) -> np.ndarray:
        """Uniformly random catalog indices, shape (n_seq, length)."""
        return rng.integers(0, len(self.cat), size=(n_seq, length))

    def sequence_chars(self, p: float, widx: np.ndarray, depths, protocol: str = "mirror",
                       xt: float = 0.0) -> np.ndarray:
        """Exact chi[seq, depth, k] (ideal SPAM) of the nested prefixes of given sequences."""
        depths = list(depths)
        mmax = max(depths)
        logc = np.log(self.factors(p, "mirror" if protocol == "mirror" else "forward", xt))
        logr = np.log(self.factors(p, "recovery", xt)) if protocol == "compiled" else None
        n_seq = widx.shape[0]
        chi = np.ones((n_seq, len(depths), self.K))
        for s in range(n_seq):
            f = self.f0
            cum = np.zeros(self.K)
            words = []
            for t in range(mmax + 1):
                if t in depths:
                    extra = 0.0
                    if protocol == "compiled" and t > 0:
                        extra = logr[self.cat.index(self.rec.word(words))]
                    chi[s, depths.index(t)] = np.exp(cum + extra)
                if t < mmax:
                    cum = cum + logc[f, widx[s, t]]
                    f = self.compose[f, widx[s, t]]
                    words.append(self.cat[widx[s, t]])
        return chi

    def experiment_dists(self, p, n_seq, depths, rng, protocol="mirror", nested=True, spam=None,
                         xt=0.0, p_mix=None) -> np.ndarray:
        """Exact P[seq, depth, outcome] of one synthetic experiment.

        nested=True: each sequence's depth-m circuit is the prefix of one
        max-depth draw (production design). nested=False: every (sequence,
        depth) circuit is an independent draw.
        spam: None (ideal) or dict(prep=bool, meas=bool, meas_scale=float).
        p_mix: optional list of (weight, p_j); the distribution is then the
        mixture over p_j (quasi-static drift: p constant within a shot,
        varying between shots), SPAM included at each p_j.
        """
        depths = list(depths)
        mmax = max(depths)
        if nested:
            widx = self.draw(n_seq, mmax, rng)
        else:
            widx = [self.draw(n_seq, mmax, rng) for _ in depths]

        def dist_at(pj):
            if nested:
                chi = self.sequence_chars(pj, widx, depths, protocol, xt)
            else:
                chi = np.concatenate([self.sequence_chars(pj, widx[i], [m], protocol, xt)
                                      for i, m in enumerate(depths)], axis=1)
            if spam is not None:
                chi = chi * self.spam(pj, **spam)
            return np.clip(T.dist_from_chars(chi, self.d), 0.0, None)

        if p_mix is None:
            return dist_at(p)
        return sum(w * dist_at(pj) for w, pj in p_mix)

    @staticmethod
    def sample_counts(P: np.ndarray, shots: int, rng) -> np.ndarray:
        """Multinomial counts with ``shots`` per (sequence, depth) circuit."""
        flat = P.reshape(-1, P.shape[-1])
        flat = flat / flat.sum(1, keepdims=True)
        return rng.multinomial(shots, flat).reshape(P.shape)
