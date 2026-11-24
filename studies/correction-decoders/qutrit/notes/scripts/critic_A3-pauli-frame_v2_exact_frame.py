#!/usr/bin/env python3
"""Critic (v2) of A3-pauli-frame-C1/C4/C5/C8: exact frame-vs-physical comparison.

Independent of the A3 code.  Random three-qutrit circuits (non-stabilizer input states: a generic
unitary on every qutrit before the first correction) with mid-circuit
Z measurements (with a classical, state-independent but asymmetric shift),
resets, stochastic Weyl noise with random NON-symmetric distributions, and a
decoder whose Weyl correction is an arbitrary (hash-like) function of the
corrected record so far.  For each circuit the EXACT joint distribution of all
recorded outcomes is computed by branching over every measurement outcome with
unnormalized density matrices, once with ideal physical corrections and once
with a Pauli frame (frame conjugated through every later Clifford via a
symplectic matrix extracted numerically from the unitaries, zeroed on reset,
X-power added to every later Z record).  The total-variation distance (TVD)
between the two joint distributions is reported.

Variants (each on the same random circuits):
  base            Weyl noise, unconditional reset to |0>                 expect 0
  reset-psi       reset to a fixed NON-stabilizer state                  expect 0
  abort           shot aborted when a corrected record satisfies a rule  expect 0
  adaptive-cliff  a Clifford applied conditionally on a corrected record
                  (frame protocol uses its real-time frame)              expect 0
  actreset-indep  active reset X^-r after reading raw r, noise always on expect 0
  actreset-dep    active reset whose conditional X gate is noisy only
                  when it fires (r != 0)                                 expect != 0
  asym-readout    level-dependent readout confusion (decay-like)         expect != 0
  coherent        one noise location replaced by a small unitary         expect != 0
  coherent-twirl  the same unitary, Weyl-twirled                         expect 0
  damping         one noise location replaced by ladder amplitude damping expect != 0
  damping-twirl   its Weyl twirl                                         expect 0
  nonclifford     H, qutrit T, H on all wires after the first correction
                  (frame follows the H gates but cannot follow T)       expect != 0
  noisy-corr      physical correction followed by its own gate noise      expect != 0

Usage: module load python/3.11; python critic_A3-pauli-frame_v2_exact_frame.py
"""

from __future__ import annotations

import itertools

import numpy as np

w = np.exp(2j * np.pi / 3)
N = 3
DIM = 3 ** N
X1 = np.zeros((3, 3), complex)
for j in range(3):
    X1[(j + 1) % 3, j] = 1
Z1 = np.diag([w ** j for j in range(3)])
H1 = np.array([[w ** (m * n) for n in range(3)] for m in range(3)]) / np.sqrt(3)
P1 = np.diag([w ** (j * (j - 1) // 2) for j in range(3)])
I3 = np.eye(3, dtype=complex)
ONE = {"H": H1, "H_INV": H1.conj().T, "P": P1, "P_INV": P1.conj().T,
       "X": X1, "Z": Z1}


def two(f):
    u = np.zeros((9, 9), complex)
    for i in range(3):
        for j in range(3):
            (o1, o2), ph = f(i, j)
            u[3 * o1 + o2, 3 * i + j] = ph
    return u


TWO = {"CNOT": two(lambda i, j: ((i, (i + j) % 3), 1)),
       "CNOT_INV": two(lambda i, j: ((i, (j - i) % 3), 1)),
       "CZ": two(lambda i, j: ((i, j), w ** (i * j))),
       "CZ_INV": two(lambda i, j: ((i, j), w ** (-i * j))),
       "SWAP": two(lambda i, j: ((j, i), 1))}

DIG = np.array(list(itertools.product(range(3), repeat=N)))   # qutrit 0 = MSB


def embed1(u, q):
    ops = [I3] * N
    ops[q] = u
    out = np.array([[1.0 + 0j]])
    for o in ops:
        out = np.kron(out, o)
    return out


def embed2(u, c, t):
    full = np.zeros((DIM, DIM), complex)
    for i_in in range(DIM):
        din = DIG[i_in]
        for oc in range(3):
            for ot in range(3):
                amp = u[3 * oc + ot, 3 * din[c] + din[t]]
                if amp == 0:
                    continue
                dout = din.copy()
                dout[c], dout[t] = oc, ot
                i_out = int(np.dot(dout, [9, 3, 1]))
                full[i_out, i_in] += amp
    return full


_WCACHE = {}


def weyl(v):
    key = tuple(int(x) % 3 for x in v)
    if key in _WCACHE:
        return _WCACHE[key]
    out = np.array([[1.0 + 0j]])
    for q in range(N):
        out = np.kron(out, np.linalg.matrix_power(X1, v[2 * q] % 3)
                      @ np.linalg.matrix_power(Z1, v[2 * q + 1] % 3))
    return out


ALLV = [tuple(v) for v in itertools.product(range(3), repeat=2 * N)]
WALL = np.array([weyl(v) for v in ALLV])
WCONJ = WALL.conj()


def weyl_id(m):
    tr = np.abs(np.einsum("vij,ij->v", WCONJ, m))
    k = int(np.argmax(tr))
    if abs(tr[k] - DIM) > 1e-8:
        return None
    return np.array(ALLV[k])


def symplectic(u):
    S = np.zeros((2 * N, 2 * N), int)
    for k in range(2 * N):
        e = np.zeros(2 * N, int)
        e[k] = 1
        img = weyl_id(u @ weyl(e) @ u.conj().T)
        if img is None:
            raise RuntimeError("not Clifford")
        S[:, k] = img
    return S


def proj(q, k):
    return np.diag((DIG[:, q] == k).astype(complex))


PROJ = {(q, k): proj(q, k) for q in range(N) for k in range(3)}


_DCACHE = {}


def dep_kraus(q, prob):
    if (q, prob) in _DCACHE:
        return _DCACHE[(q, prob)]
    ks = [np.sqrt(1 - prob) * np.eye(DIM)]
    for a, b in itertools.product(range(3), repeat=2):
        if (a, b) != (0, 0):
            v = [0] * (2 * N)
            v[2 * q], v[2 * q + 1] = a, b
            ks.append(np.sqrt(prob / 8) * weyl(v))
    _DCACHE[(q, prob)] = ks
    return ks


def apply_kraus(rho, ks):
    return sum(k @ rho @ k.conj().T for k in ks)


def twirl_kraus_1q(ks1, q):
    """Weyl twirl of a one-qutrit channel given by Kraus ops, as a Pauli channel."""
    # chi-diagonal: p(a,b) = sum_K |tr(W(a,b)^dag K)|^2 / 9
    probs = {}
    for a, b in itertools.product(range(3), repeat=2):
        Wab = np.linalg.matrix_power(X1, a) @ np.linalg.matrix_power(Z1, b)
        probs[(a, b)] = sum(abs(np.trace(Wab.conj().T @ k)) ** 2 for k in ks1) / 9
    out = []
    for (a, b), p in probs.items():
        v = [0] * (2 * N)
        v[2 * q], v[2 * q + 1] = a, b
        out.append(np.sqrt(p) * weyl(v))
    return out


ADAPT_U = (embed1(H1, 2), symplectic(embed1(H1, 2)))
_T1 = np.diag([1, np.exp(2j * np.pi / 9), np.exp(-2j * np.pi / 9)])
T_ALL = np.kron(np.kron(_T1, _T1), _T1)
_HH = np.kron(np.kron(H1, H1), H1)
H_ALL = (_HH, symplectic(_HH))

# ---------------------------------------------------------------- circuits

def random_circuit(seed):
    rng = np.random.default_rng(seed)
    ops = []

    def rand_cliff():
        if rng.random() < 0.5:
            name = rng.choice(list(ONE))
            q = int(rng.integers(N))
            u = embed1(ONE[name], q)
        else:
            name = rng.choice(list(TWO))
            c, t = rng.choice(N, 2, replace=False)
            u = embed2(TWO[name], int(c), int(t))
        return ("U", u, symplectic(u))

    def rand_noise():
        k = int(rng.integers(1, 4))
        vs = [tuple(int(x) for x in rng.integers(0, 3, 2 * N)) for _ in range(k)]
        ps = rng.dirichlet(np.ones(k)) * rng.uniform(0.05, 0.25)
        return ("NW", list(zip(ps, vs)))

    # a generic (non-Clifford) unitary on every qutrit BEFORE the first
    # correction: inputs are non-stabilizer states (allowed by A1)
    v0 = np.array([[1.0 + 0j]])
    for _ in range(N):
        gm = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
        qm, _r = np.linalg.qr(gm)
        v0 = np.kron(v0, qm)
    ops.append(("U0", v0))
    xi = (0.93, 0.05, 0.02)                    # asymmetric classical shift
    layout = ["C", "C", "N", "M0", "D", "C", "N", "C", "M1", "D", "R1",
              "C", "N", "C", "C", "M2", "D", "C", "N", "MF"]
    for item in layout:
        if item == "C":
            ops.append(rand_cliff())
        elif item == "N":
            ops.append(rand_noise())
        elif item.startswith("M") and item != "MF":
            ops.append(("M", int(item[1]), xi))
        elif item == "D":
            ops.append(("D", len(ops)))
        elif item == "R1":
            ops.append(("R", 1, None))
        elif item == "MF":
            for q in range(N):
                ops.append(("M", q, xi))
    return ops


def decoder(record, step):
    seed = 1000003 * step + sum((r + 1) * 7 ** i for i, r in enumerate(record))
    g = np.random.default_rng(seed)
    if g.random() < 0.2:
        return np.zeros(2 * N, int)
    return g.integers(0, 3, 2 * N)


# ---------------------------------------------------------------- simulator

def simulate(ops, protocol, opts):
    out = {}

    def rec(i, rho, f, record, corrected_once):
        tr = np.real(np.trace(rho))
        if tr < 1e-15:
            return
        if i == len(ops):
            out[tuple(record)] = out.get(tuple(record), 0.0) + tr
            return
        op = ops[i]
        kind = op[0]
        if kind == "U":
            rho = op[1] @ rho @ op[1].conj().T
            if protocol == "frame":
                f = (op[2] @ f) % 3
            if opts.get("nonclifford") and i == opts["t_at"]:
                # H on every qutrit (Clifford: frame follows), qutrit T gate on
                # every qutrit (frame cannot follow it: left unchanged), then H
                # on every qutrit again (frame follows)
                rho = H_ALL[0] @ rho @ H_ALL[0].conj().T
                if protocol == "frame":
                    f = (H_ALL[1] @ f) % 3
                rho = T_ALL @ rho @ T_ALL.conj().T
                rho = H_ALL[0] @ rho @ H_ALL[0].conj().T
                if protocol == "frame":
                    f = (H_ALL[1] @ f) % 3
            return rec(i + 1, rho, f, record, corrected_once)
        if kind == "U0":
            if np.any(f):
                raise RuntimeError("non-Clifford U0 after a correction")
            rho = op[1] @ rho @ op[1].conj().T
            return rec(i + 1, rho, f, record, corrected_once)
        if kind == "NW":
            sub = opts.get("replace", {}).get(i)
            if sub is not None:
                return rec(i + 1, apply_kraus(rho, sub), f, record, corrected_once)
            p0 = 1 - sum(p for p, _ in op[1])
            new = p0 * rho
            for p, v in op[1]:
                Wv = weyl(v)
                new = new + p * (Wv @ rho @ Wv.conj().T)
            return rec(i + 1, new, f, record, corrected_once)
        if kind == "M":
            q, xi = op[1], op[2]
            a = int(f[2 * q]) if protocol == "frame" else 0
            conf = opts.get("confusion")
            for k in range(3):                 # recorded (corrected) outcome
                new = np.zeros_like(rho)
                for raw in range(3):
                    if conf is not None:      # P(recorded raw' | raw), level dependent
                        pr = conf[(k - a) % 3, raw]
                    else:
                        pr = xi[(k - a - raw) % 3]
                    if pr == 0:
                        continue
                    Pm = PROJ[(q, raw)]
                    new = new + pr * (Pm @ rho @ Pm)
                rec_k = record + [k]
                if opts.get("abort") and len(rec_k) == 2 and rec_k[-1] != 0:
                    tr2 = np.real(np.trace(new))
                    if tr2 > 1e-15:
                        key = tuple(rec_k + ["ABORT"])
                        out[key] = out.get(key, 0.0) + tr2
                    continue
                if opts.get("adaptive") and len(rec_k) == 1 and rec_k[0] == 1:
                    u, Su = ADAPT_U
                    new = u @ new @ u.conj().T
                    f2 = (Su @ f) % 3 if protocol == "frame" else f
                    rec(i + 1, new, f2, rec_k, corrected_once)
                    continue
                rec(i + 1, new, f, rec_k, corrected_once)
            return
        if kind == "R":
            q = op[1]
            act = opts.get("actreset")
            if act is None:
                psi = opts.get("reset_state")
                if psi is None:
                    psi = np.array([1, 0, 0], complex)
                ks = []
                for j in range(3):
                    k1 = np.outer(psi, np.eye(3)[j])
                    ks.append(embed1(k1, q))
                rho = apply_kraus(rho, ks)
            else:
                # measure raw r, apply X^{-r}; gate noise when it fires (dep) or always (indep)
                new = np.zeros_like(rho)
                for r in range(3):
                    Pm = PROJ[(q, r)]
                    xr = embed1(np.linalg.matrix_power(X1, (-r) % 3), q)
                    br = xr @ Pm @ rho @ Pm @ xr.conj().T
                    if act == "indep" or (act == "dep" and r != 0):
                        br = apply_kraus(br, dep_kraus(q, 0.1))
                    new = new + br
                rho = new
            if protocol == "frame":
                f = f.copy()
                f[2 * q] = f[2 * q + 1] = 0
            return rec(i + 1, rho, f, record, corrected_once)
        if kind == "D":
            c = decoder(record, op[1])
            nontriv = bool(np.any(c))
            if protocol == "physical":
                Wc = weyl(c)
                rho = Wc @ rho @ Wc.conj().T
                if opts.get("noisy_corr") and nontriv:
                    for q in range(N):
                        if c[2 * q] or c[2 * q + 1]:
                            rho = apply_kraus(rho, dep_kraus(q, 0.05))
            else:
                f = (f + c) % 3
            return rec(i + 1, rho, f, record, corrected_once or nontriv)
        raise KeyError(kind)

    rho0 = np.zeros((DIM, DIM), complex)
    rho0[0, 0] = 1
    rec(0, rho0, np.zeros(2 * N, int), [], False)
    return out


def tvd(d1, d2):
    keys = set(d1) | set(d2)
    return 0.5 * sum(abs(d1.get(k, 0.0) - d2.get(k, 0.0)) for k in keys)


def main():
    seeds = [11, 12, 13, 14, 15]
    th = 0.25
    rngu = np.random.default_rng(5)
    g = rngu.normal(size=(3, 3)) + 1j * rngu.normal(size=(3, 3))
    g = (g + g.conj().T) / 2
    ev, evec = np.linalg.eigh(g)
    V1 = evec @ np.diag(np.exp(-1j * th * ev)) @ evec.conj().T
    gam = 0.15
    AD = [np.diag([1, np.sqrt(1 - gam), np.sqrt(1 - 2 * gam)]).astype(complex),
          np.array([[0, np.sqrt(gam), 0], [0, 0, 0], [0, 0, 0]], complex),
          np.array([[0, 0, 0], [0, 0, np.sqrt(2 * gam)], [0, 0, 0]], complex)]
    conf = np.array([[1.0, 0.06, 0.02],      # rows: recorded, cols: raw level
                     [0.0, 0.94, 0.08],
                     [0.0, 0.00, 0.90]])
    psi = np.array([0.8, 0.36 + 0.3j, 0.2 - 0.3j])
    psi = psi / np.linalg.norm(psi)
    print("TVD between the exact joint record distributions, physical vs frame")
    print(f"{'variant':16s} " + " ".join(f"seed{s:>3d}" for s in seeds))
    variants = ["base", "reset-psi", "abort", "adaptive-cliff", "actreset-indep",
                "actreset-dep", "asym-readout", "coherent", "coherent-twirl",
                "damping", "damping-twirl", "nonclifford", "noisy-corr"]
    for var in variants:
        row = []
        for s in seeds:
            ops = random_circuit(s)
            noise_idx = [i for i, o in enumerate(ops) if o[0] == "NW"]
            d_idx = [i for i, o in enumerate(ops) if o[0] == "D"]
            j = noise_idx[-1]                       # a noise location after corrections
            qn = 0
            opts = {}
            if var == "reset-psi":
                opts["reset_state"] = psi
            elif var == "abort":
                opts["abort"] = True
            elif var == "adaptive-cliff":
                opts["adaptive"] = True
            elif var == "actreset-indep":
                opts["actreset"] = "indep"
            elif var == "actreset-dep":
                opts["actreset"] = "dep"
            elif var == "asym-readout":
                opts["confusion"] = conf
            elif var == "coherent":
                opts["replace"] = {j: [embed1(V1, qn)]}
            elif var == "coherent-twirl":
                opts["replace"] = {j: twirl_kraus_1q([V1], qn)}
            elif var == "damping":
                opts["replace"] = {j: [embed1(k, qn) for k in AD]}
            elif var == "damping-twirl":
                opts["replace"] = {j: twirl_kraus_1q(AD, qn)}
            elif var == "nonclifford":
                u_after = [i for i, o in enumerate(ops) if o[0] == "U" and i > d_idx[0]]
                opts["nonclifford"] = True
                opts["t_at"] = u_after[0]
            elif var == "noisy-corr":
                opts["noisy_corr"] = True
            dp = simulate(ops, "physical", opts)
            df = simulate(ops, "frame", opts)
            assert abs(sum(dp.values()) - 1) < 1e-9 and abs(sum(df.values()) - 1) < 1e-9
            row.append(tvd(dp, df))
        print(f"{var:16s} " + " ".join(f"{x:7.1e}" for x in row))


if __name__ == "__main__":
    main()
