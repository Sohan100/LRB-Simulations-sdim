#!/usr/bin/env python3
"""A4 toy 2: hidden-Markov model of LRB-D vs correction for [[5,1,2]]_3.

Phenomenological caricature of the LRB-D circuit (A4-nonmarkov-theory.md
section 4).  Hidden state after each step: s = (sigma, b) in Z_3^5, where
sigma in Z_3^4 is the syndrome of the data Pauli frame f (sigma_i = <S_i, f>,
symplectic form <A,B> = z_A.x_B - x_A.z_B) and b = z_0 + z_3 is the shift of
the ideal terminal readout x0 + x3.  The data noise is i.i.d. single-qutrit
depolarising (prob q per qutrit per step, uniform over the 8 non-identity
Weyl operators), the syndrome read-out has i.i.d. shift errors (prob q_m per
stabiliser), logical gates are omitted (identity sequence: this isolates the
memory mechanisms; the Clifford twirl is treated in section 2 of the notes).

A depth-m "LRB-D-like" run is: m forward steps, each followed by a check
round (data noise q_c during the round, read-out noise q_m), the check is
used or not according to the policy; then m unchecked "inverse" steps (data
noise q_i); then a final noisy check; then ideal read-out of b.
Protocols:
  det-unif-k   : postselect on used rounds (every k-th) and the final round;
  det-const-1  : postselect on the final round only;
  cor-unif-k   : on used rounds apply a fixed minimum-weight lookup
                 correction (memoryless decoder), final round decoded by ML;
  cor-const-1  : no intermediate action, final round decoded by ML.
Everything is exact (243-state transfer matrices, Fourier convolution).
Also prints the spectrum of the one-step transfer operator of the forward
block, which sets the memory time (A4 notes, section 4.2).

    module load python/3.11
    python docs/correction-decoders/qutrit-512/notes/scripts/A4_hmm_toy.py
"""
from __future__ import annotations

import itertools

import numpy as np

w = np.exp(2j * np.pi / 3)
n = 5
STABS = [  # (x vector, z vector)
    ((0, 0, 0, 0, 0), (1, 2, 0, 2, 0)),   # S_Z1 = Z0 Z1^-1 Z3^-1
    ((0, 0, 0, 0, 0), (0, 1, 1, 0, 2)),   # S_Z2 = Z1 Z2 Z4^-1
    ((1, 1, 2, 0, 0), (0, 0, 0, 0, 0)),   # S_X1 = X0 X1 X2^-1
    ((0, 2, 0, 1, 2), (0, 0, 0, 0, 0)),   # S_X2 = X1^-1 X3 X4^-1
]
G = np.array(list(itertools.product(range(3), repeat=5)))  # states (243, 5)
NG = len(G)
IDX = {tuple(g): i for i, g in enumerate(G)}
CHAR = w ** (-(G @ G.T % 3))                               # (chi, g)


def image(x: np.ndarray, z: np.ndarray) -> tuple:
    sig = [int((np.dot(sz, x) - np.dot(sx, z)) % 3) for sx, sz in
           [(np.array(a), np.array(b)) for a, b in STABS]]
    return tuple(sig + [int((z[0] + z[3]) % 3)])


def step_distribution(q: float) -> np.ndarray:
    """Distribution on Z_3^5 of one step of i.i.d. depolarising data noise."""
    P = np.zeros(NG)
    P[IDX[(0,) * 5]] = 1.0
    for qubit in range(n):
        single = np.zeros(NG)
        for a in range(3):
            for c in range(3):
                x = np.zeros(n, dtype=int)
                z = np.zeros(n, dtype=int)
                x[qubit], z[qubit] = a, c
                pr = (1 - q) if (a, c) == (0, 0) else q / 8
                single[IDX[image(x, z)]] += pr
        P = convolve(P, single)
    return P


def convolve(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    out = np.conj(CHAR).T @ ((CHAR @ P) * (CHAR @ Q)) / NG
    return out.real


def meas_dist(qm: float) -> np.ndarray:
    """P(mu) for mu in Z_3^4, independent shifts per stabiliser."""
    mus = np.array(list(itertools.product(range(3), repeat=4)))
    return np.prod(np.where(mus == 0, 1 - qm, qm / 2), axis=1), mus


SYN = np.array(list(itertools.product(range(3), repeat=4)))
SIDX = {tuple(s): i for i, s in enumerate(SYN)}


def state_index(sig, b) -> int:
    return IDX[tuple(list(sig) + [b])]


def lookup_decoder(q: float) -> np.ndarray:
    """b-correction for each syndrome from the most likely single step error.

    b_hat(s) = argmax_b P_step(s, b); ties broken towards the smallest b
    (this is where distance 2 bites: Z0 and Z2^-1 share a syndrome but
    differ by Z_L)."""
    P = step_distribution(q)
    bhat = np.zeros(81, dtype=int)
    for i, s in enumerate(SYN):
        probs = [P[state_index(s, b)] for b in range(3)]
        bhat[i] = int(np.argmax(np.round(probs, 15)))
    return bhat


def check_detect(P: np.ndarray, qm: float) -> np.ndarray:
    """Postselect: accept iff sigma + mu = 0 (masked errors persist)."""
    pm, mus = meas_dist(qm)
    out = P.copy()
    for i, g in enumerate(G):
        sig = g[:4]
        out[i] *= pm[SIDX[tuple((-sig) % 3)]]
    return out


def check_correct(P: np.ndarray, qm: float, bhat: np.ndarray) -> np.ndarray:
    """Observe s = sigma + mu, apply correction with syndrome s and b-part
    bhat[s]; residual syndrome is -mu."""
    pm, mus = meas_dist(qm)
    out = np.zeros(NG)
    for i, g in enumerate(G):
        if P[i] == 0:
            continue
        sig, b = g[:4], g[4]
        for j, mu in enumerate(mus):
            s = (sig + mu) % 3
            new = state_index((-mu) % 3, (b - bhat[SIDX[tuple(s)]]) % 3)
            out[new] += P[i] * pm[j]
    return out


def final_ml(P: np.ndarray, qm: float) -> tuple[float, float]:
    """Final noisy check decoded by maximum likelihood (genie posterior)."""
    pm, mus = meas_dist(qm)
    joint = np.zeros((81, 3))                 # P(s_obs, b)
    for i, g in enumerate(G):
        sig, b = g[:4], g[4]
        for j, mu in enumerate(mus):
            joint[SIDX[tuple((sig + mu) % 3)], b] += P[i] * pm[j]
    num = 0
    for si, s in enumerate(SYN):
        # residual b after correcting with the most likely class
        bh = int(np.argmax(joint[si]))
        num += sum(joint[si, b] * w ** ((b - bh) % 3) for b in range(3))
    tot = joint.sum()
    return tot, abs(num) / tot


def readout(P: np.ndarray) -> tuple[float, float]:
    tot = P.sum()
    bs = G[:, 4]
    return tot, abs(np.sum(P * w ** bs)) / tot


def run(policy: str, k: int, m_max: int, q_c: float, q_m: float,
        q_i: float, bhat: np.ndarray) -> np.ndarray:
    """Return array (m, acceptance, fidelity) for m = 0..m_max."""
    mu_c = step_distribution(q_c)
    mu_half = step_distribution(q_c / 2)   # final round: half before/after
    mu_i = step_distribution(q_i)
    hat_i = CHAR @ mu_i
    res = []
    P = np.zeros(NG)
    P[IDX[(0,) * 5]] = 1.0
    fwd = [P.copy()]
    for t in range(1, m_max + 1):         # forward block, step t
        P = convolve(P, mu_c)
        used = (k > 0) and (t % k == 0)
        if used:
            if policy == "det":
                P = check_detect(P, q_m)
            else:
                P = check_correct(P, q_m, bhat)
        fwd.append(P.copy())
    for m in range(0, m_max + 1):
        # inverse block: m unchecked steps, then the final round
        Pi = (np.conj(CHAR).T @ ((CHAR @ fwd[m]) * hat_i ** m) / NG).real
        Pi = convolve(Pi, mu_half)        # data noise seen by final check
        if policy == "det":
            Pi = check_detect(Pi, q_m)
            Pi = convolve(Pi, mu_half)    # data noise after the final check
            acc, F = readout(Pi)
        else:
            # errors after the final check are not seen by the decoder:
            # decode the pre-check state, then add the late noise to b only
            tot, F_dec = final_ml(Pi, q_m)
            F_late = readout(mu_half)[1]  # character of late noise on b
            acc, F = tot, F_dec * F_late
        res.append((m, acc, F))
    return np.array(res)


def window_rates(F: np.ndarray, ms: np.ndarray) -> np.ndarray:
    return (F[1:] / F[:-1]) ** (1 / np.diff(ms))


def best_single_exp(ms, F):
    A = np.stack([np.ones_like(ms, dtype=float), ms], 1)
    beta, *_ = np.linalg.lstsq(A, np.log(F), rcond=None)
    fit = np.exp(A @ beta)
    return np.exp(beta[1]), np.max(np.abs(F / fit - 1))


def forward_operator(policy: str, q_c: float, q_m: float, bhat) -> np.ndarray:
    T = np.zeros((NG, NG))
    mu_c = step_distribution(q_c)
    for j in range(NG):
        e = np.zeros(NG)
        e[j] = 1
        e = convolve(e, mu_c)
        e = check_detect(e, q_m) if policy == "det" else \
            check_correct(e, q_m, bhat)
        T[:, j] = e
    return T


def main() -> None:
    m_max = 24
    ms = np.arange(0, m_max + 1)
    for p in (1e-3, 5e-3):
        q_c, q_m, q_i = 12 * p, 7 * p, 3 * p   # SI1000-like per step
        bhat = lookup_decoder(q_c)
        print(f"\n=== p = {p}: q_check = {q_c}, q_meas = {q_m}, "
              f"q_inverse = {q_i} ===")
        print(" protocol       F(0)      F(2)      F(12)     F(24)    "
              "acc(24)   rate(2->4) rate(22->24) single-exp max dev")
        for pol, k, lab in [("det", 1, "det-unif-1"), ("det", 4, "det-unif-4"),
                            ("det", 0, "det-const-1"),
                            ("cor", 1, "cor-unif-1"), ("cor", 4, "cor-unif-4"),
                            ("cor", 0, "cor-const-1")]:
            r = run(pol, k, m_max, q_c, q_m, q_i, bhat)
            F, acc = r[:, 2], r[:, 1]
            rates = window_rates(F[2:], ms[2:])
            f1, dev = best_single_exp(ms[2:], F[2:])
            print(f" {lab:12s} {F[0]:.6f}  {F[2]:.6f}  {F[12]:.6f}  "
                  f"{F[24]:.6f}  {acc[24]:.4f}   {rates[0]:.6f}   "
                  f"{rates[-1]:.6f}   {dev:.2e}")
        # spectrum of the forward one-step operator (check every step)
        for pol in ("det", "cor"):
            T = forward_operator(pol, q_c, q_m, bhat)
            ev = np.linalg.eigvals(T)
            ev = ev[np.argsort(-np.abs(ev))]
            # modes that carry the readout character (b-sector)
            vals, vecs = np.linalg.eig(T)
            left = np.linalg.inv(vecs)
            bchar = w ** G[:, 4]
            start = np.zeros(NG)
            start[IDX[(0,) * 5]] = 1
            weight = np.abs((bchar @ vecs) * (left @ start))
            order = np.argsort(-np.abs(vals))
            sig = [(abs(vals[i]), weight[i]) for i in order
                   if weight[i] > 1e-12][:4]
            print(f" forward one-step operator ({pol}): |lambda| top 4 = "
                  f"{np.round(np.abs(ev[:4]), 6)}; readout-carrying modes "
                  f"(|lambda|, weight) = "
                  f"{[(round(a, 6), float('%.1e' % b)) for a, b in sig]}")


if __name__ == "__main__":
    main()
