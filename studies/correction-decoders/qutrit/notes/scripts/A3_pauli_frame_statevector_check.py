#!/usr/bin/env python3
"""A3: dense state-vector check of Pauli-frame correction versus physical correction.

Independent of sdim's tableau/DEM machinery: the circuit text files (.chp) of the
baseline run are parsed directly and simulated with dense 3^9 state vectors using
the sdim gate conventions (X|j>=|j+1>, Z=diag(w^j), H=Fourier, P=diag(w^{j(j-1)/2}),
CNOT|i,j>=|i,i+j>, CZ=diag(w^{ij}), SWAP standard). The baseline run folder is only
read, never written.

What it checks
  1. Gate table: every gate used maps Weyl operators to Weyl operators (up to phase);
     the symplectic image of each generator is printed (the frame-update rule).
  2. Noiseless circuit: every ancilla outcome and the logical outcome x0+x3 are
     deterministic and equal to 0 (validates conventions).
  3. Pauli mode (the circuit's own SI1000 noise, optionally rescaled): a real-time
     lookup decoder acts after every check round.  Protocol "physical" applies the
     correction to the state vector (ideal, instantaneous).  Protocol "frame" applies
     nothing, tracks the frame (conjugated by every later Clifford, cleared by reset)
     and corrects every later measurement outcome.  With common random numbers
     (same noise draws, same uniform u per measurement) the corrected records must
     agree shot by shot.
  4. Coherent mode: every data-qutrit N1 'd' location is replaced by the fixed
     unitary V = diag(1, e^{i th}, e^{2 i th}); measurement-flip N1 'f' locations are
     kept stochastic.  Same two protocols; the exact logical-outcome distribution of
     each shot is computed from the final state and the paired difference of the
     two protocols is reported (per-shot distributions differ; the Clifford- and
     shot-averaged difference was not resolved with 150 shots at theta = 0.1).
  5. Minimal mode: exact one-qutrit density-matrix examples (error E, noise,
     correction C = E^-1 physical or in the frame, noise, readout) for a coherent
     Z rotation, a generic unitary, amplitude damping, and their Weyl twirls.

Usage (login node, a few seconds to ~2 minutes):
  module load python/3.11
  python A3_pauli_frame_statevector_check.py --mode all
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import itertools
import time

import numpy as np

D = 3
W = np.exp(2j * np.pi / D)
NQ = 9
DATA = (0, 1, 2, 3, 4)
ANCILLAS = (5, 6, 7, 8)
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")

# ----------------------------------------------------------------------------
# gate matrices (sdim conventions)
X1 = np.zeros((3, 3), complex)
for _j in range(3):
    X1[(_j + 1) % 3, _j] = 1.0
Z1 = np.diag([W ** j for j in range(3)])
H1 = np.array([[W ** (m * n) for n in range(3)] for m in range(3)]) / np.sqrt(3)
P1 = np.diag([W ** (j * (j - 1) // 2) for j in range(3)])
I3 = np.eye(3)


def _two(f):
    u = np.zeros((9, 9), complex)
    for i in range(3):
        for j in range(3):
            out, ph = f(i, j)
            u[3 * out[0] + out[1], 3 * i + j] = ph
    return u


CNOT = _two(lambda i, j: ((i, (i + j) % 3), 1.0))
CNOT_INV = _two(lambda i, j: ((i, (j - i) % 3), 1.0))
CZ = _two(lambda i, j: ((i, j), W ** (i * j)))
CZ_INV = _two(lambda i, j: ((i, j), W ** (-i * j)))
SWAP = _two(lambda i, j: ((j, i), 1.0))

ONE_Q = {
    "I": I3, "X": X1, "X_INV": X1.conj().T, "Z": Z1, "Z_INV": Z1.conj().T,
    "H": H1, "H_INV": H1.conj().T, "P": P1, "P_INV": P1.conj().T,
}
TWO_Q = {"CNOT": CNOT, "CNOT_INV": CNOT_INV, "CZ": CZ, "CZ_INV": CZ_INV,
         "SWAP": SWAP}


def weyl(a, b):
    return np.linalg.matrix_power(X1, a % 3) @ np.linalg.matrix_power(Z1, b % 3)


def identify_weyl(m, nq):
    """Return Weyl vector (a1,b1,...) with m proportional to W(v), else None."""
    for v in itertools.product(range(3), repeat=2 * nq):
        w = np.array([[1.0]])
        for q in range(nq):
            w = np.kron(w, weyl(v[2 * q], v[2 * q + 1]))
        k = np.argmax(np.abs(w.flatten()))
        c = m.flatten()[k] / w.flatten()[k]
        if abs(abs(c) - 1) < 1e-9 and np.allclose(m, c * w, atol=1e-9):
            return v
    return None


def symplectic_table():
    """Images of the Weyl generators under conjugation U . U^dagger."""
    table = {}
    for name, u in ONE_Q.items():
        imgs = []
        for gen in (X1, Z1):
            v = identify_weyl(u @ gen @ u.conj().T, 1)
            if v is None:
                raise RuntimeError(f"{name} is not Clifford")
            imgs.append(v)
        table[name] = imgs  # [image of X, image of Z], each (a,b)
    for name, u in TWO_Q.items():
        imgs = []
        for gen in (np.kron(X1, I3), np.kron(Z1, I3), np.kron(I3, X1),
                    np.kron(I3, Z1)):
            v = identify_weyl(u @ gen @ u.conj().T, 2)
            if v is None:
                raise RuntimeError(f"{name} is not Clifford")
            imgs.append(v)
        table[name] = imgs  # images of X_c, Z_c, X_t, Z_t as (a1,b1,a2,b2)
    return table


SYMP = None


def frame_update(frame, name, args):
    """frame: int array shape (NQ, 2) of (a, b) powers; conjugate by the gate."""
    if name in ONE_Q:
        q = args[0]
        a, b = frame[q]
        ix, iz = SYMP[name]
        frame[q] = ((a * ix[0] + b * iz[0]) % 3, (a * ix[1] + b * iz[1]) % 3)
        return
    if name in TWO_Q:
        c, t = args
        ac, bc = frame[c]
        at, bt = frame[t]
        gx_c, gz_c, gx_t, gz_t = SYMP[name]
        new = np.zeros(4, int)
        for coef, img in ((ac, gx_c), (bc, gz_c), (at, gx_t), (bt, gz_t)):
            new = (new + coef * np.array(img)) % 3
        frame[c] = (new[0], new[1])
        frame[t] = (new[2], new[3])
        return
    raise KeyError(name)


# ----------------------------------------------------------------------------
# circuit parsing

def parse_chp(path):
    ops = []
    with open(path) as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            toks = line.split()
            name = toks[0]
            if name in ("Randomly-generated", "d", "DETECTOR",
                        "LOGICAL_OBSERVABLE", "TICK"):
                continue
            args, params = [], {}
            for t in toks[1:]:
                if "=" in t:
                    k, v = t.split("=", 1)
                    params[k] = v.strip('"')
                else:
                    args.append(int(t))
            ops.append((name, tuple(args), params))
    return ops


def round_end_indices(ops):
    """Indices of the measurement op that closes each check round (M on wire 8)."""
    return [i for i, (n, a, _) in enumerate(ops) if n == "M" and a[0] == 8]


# ----------------------------------------------------------------------------
# dense simulator

DIM = 3 ** NQ
_IDX = np.arange(DIM)
STRIDE = [3 ** (NQ - 1 - q) for q in range(NQ)]
DIG = [(_IDX // STRIDE[q]) % 3 for q in range(NQ)]
_CACHE: dict = {}


def _perm_x(q, a):
    key = ("x", q, a % 3)
    if key not in _CACHE:
        _CACHE[key] = _IDX + (((DIG[q] - a) % 3) - DIG[q]) * STRIDE[q]
    return _CACHE[key]


def _phase_z(q, b):
    key = ("z", q, b % 3)
    if key not in _CACHE:
        _CACHE[key] = W ** ((b * DIG[q]) % 3)
    return _CACHE[key]


def _two_q_action(name, c, t):
    """Permutation (gather index) or phase vector for a two-qutrit gate."""
    key = (name, c, t)
    if key in _CACHE:
        return _CACHE[key]
    dc, dt = DIG[c], DIG[t]
    if name == "CNOT":        # |i,j> -> |i,i+j| : new[i,j] = old[i,j-i]
        val = ("perm", _IDX + (((dt - dc) % 3) - dt) * STRIDE[t])
    elif name == "CNOT_INV":  # new[i,j] = old[i,j+i]
        val = ("perm", _IDX + (((dt + dc) % 3) - dt) * STRIDE[t])
    elif name == "SWAP":      # new[i,j] = old[j,i]
        val = ("perm", _IDX + (dt - dc) * STRIDE[c] + (dc - dt) * STRIDE[t])
    elif name == "CZ":
        val = ("phase", W ** ((dc * dt) % 3))
    elif name == "CZ_INV":
        val = ("phase", W ** ((-dc * dt) % 3))
    else:
        raise KeyError(name)
    _CACHE[key] = val
    return val


class Dense:
    """Flat 3^9 state vector; qutrit q is digit q (most significant first)."""

    def __init__(self):
        self.psi = np.zeros(DIM, complex)
        self.psi[0] = 1.0

    def apply1(self, q, u):
        if np.allclose(u, np.diag(np.diag(u))):
            self.psi = self.psi * np.diag(u)[DIG[q]]
            return
        v = self.psi.reshape(3 ** q, 3, 3 ** (NQ - 1 - q))
        self.psi = np.einsum("ij,ajb->aib", u, v).reshape(DIM)

    def apply_named1(self, q, name):
        if name in ("I",):
            return
        if name == "X":
            self.psi = self.psi[_perm_x(q, 1)]
        elif name == "X_INV":
            self.psi = self.psi[_perm_x(q, 2)]
        elif name == "Z":
            self.psi = self.psi * _phase_z(q, 1)
        elif name == "Z_INV":
            self.psi = self.psi * _phase_z(q, 2)
        else:
            self.apply1(q, ONE_Q[name])

    def apply_named2(self, c, t, name):
        kind, arr = _two_q_action(name, c, t)
        if kind == "perm":
            self.psi = self.psi[arr]
        else:
            self.psi = self.psi * arr

    def weyl(self, q, a, b):
        a %= 3
        b %= 3
        if b:
            self.psi = self.psi * _phase_z(q, b)
        if a:
            self.psi = self.psi[_perm_x(q, a)]

    def probs(self, q):
        p = np.sum((np.abs(self.psi) ** 2).reshape(3 ** q, 3, 3 ** (NQ - 1 - q)),
                   axis=(0, 2))
        return p / p.sum()

    def project(self, q, k):
        self.psi = np.where(DIG[q] == k, self.psi, 0.0)
        self.psi /= np.linalg.norm(self.psi)

    def pair_marginal(self, q1, q2):
        p = (np.abs(self.psi) ** 2)
        out = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                out[i, j] = p[(DIG[q1] == i) & (DIG[q2] == j)].sum()
        return out


def sample_cdf(p, u):
    c = np.cumsum(p)
    k = int(np.searchsorted(c, u * c[-1], side="right"))
    return min(k, 2)


# ----------------------------------------------------------------------------
# noise draws (common random numbers)

NONZERO2 = [v for v in itertools.product(range(3), repeat=2) if any(v)]
NONZERO4 = [v for v in itertools.product(range(3), repeat=4) if any(v)]


def draw_noise(ops, rng, p_base):
    """Per op: None (no fault) or a Weyl vector; per M/RESET: a uniform u."""
    draws = []
    for name, args, params in ops:
        if name == "N1":
            prob = float(params["prob"]) if p_base is None else \
                float(params["prob_scale"]) * p_base
            ch = params.get("noise_channel", params.get("channel", "d"))
            if rng.random() < prob:
                if ch == "d":
                    draws.append(NONZERO2[rng.integers(len(NONZERO2))])
                elif ch == "f":
                    draws.append((int(rng.integers(1, 3)), 0))
                elif ch == "p":
                    draws.append((0, int(rng.integers(1, 3))))
                else:
                    raise ValueError(ch)
            else:
                draws.append(None)
        elif name == "N2":
            prob = float(params["prob"]) if p_base is None else \
                float(params["prob_scale"]) * p_base
            if rng.random() < prob:
                draws.append(NONZERO4[rng.integers(len(NONZERO4))])
            else:
                draws.append(None)
        elif name in ("M", "RESET"):
            draws.append(rng.random())
        else:
            draws.append(None)
    return draws


# ----------------------------------------------------------------------------
# lookup decoder from weight-1 data errors (built by frame propagation)

def build_lookup(ops):
    ends = round_end_indices(ops)
    first_end = ends[0]
    start = min(i for i, (n, a, _) in enumerate(ops)
                if n in TWO_Q and any(x in ANCILLAS for x in a))
    seg = ops[start:first_end + 1]
    table = {}
    for q in DATA:
        for (a, b) in NONZERO2:
            fr = np.zeros((NQ, 2), int)
            fr[q] = (a, b)
            synd = []
            for name, args, _ in seg:
                if name in ONE_Q or name in TWO_Q:
                    frame_update(fr, name, args)
                elif name == "M":
                    synd.append(int(fr[args[0]][0]))
            synd = tuple(synd)
            if synd not in table and any(synd):
                corr = np.zeros((NQ, 2), int)
                corr[q] = ((-a) % 3, (-b) % 3)
                table[synd] = corr
    return table


# ----------------------------------------------------------------------------
# one shot, two protocols

def run_shot(ops, draws, lookup, protocol, coherent_theta=None,
             exact_logical=False):
    sim = Dense()
    frame = np.zeros((NQ, 2), int)
    ends = set(round_end_indices(ops))
    record = []
    round_buf = []
    vtheta = None
    if coherent_theta is not None:
        vtheta = np.diag([np.exp(1j * coherent_theta * j) for j in range(3)])
    n_ops = len(ops)
    term_meas = [i for i, (n, a, _) in enumerate(ops) if n == "M" and a[0] in (0, 3)]
    for i, (name, args, params) in enumerate(ops):
        if exact_logical and i == term_meas[0]:
            break
        if name in ONE_Q:
            sim.apply_named1(args[0], name)
            if protocol == "frame":
                frame_update(frame, name, args)
        elif name in TWO_Q:
            sim.apply_named2(args[0], args[1], name)
            if protocol == "frame":
                frame_update(frame, name, args)
        elif name == "N1":
            q = args[0]
            ch = params.get("noise_channel", params.get("channel", "d"))
            if vtheta is not None and ch == "d" and q in DATA:
                sim.apply1(q, vtheta)          # coherent error, always on
            elif vtheta is not None and ch == "d":
                pass                           # ancilla depolarizing off
            elif draws[i] is not None:
                sim.weyl(q, *draws[i])
        elif name == "N2":
            if vtheta is not None:
                pass                           # two-qutrit noise off
            elif draws[i] is not None:
                a1, b1, a2, b2 = draws[i]
                sim.weyl(args[0], a1, b1)
                sim.weyl(args[1], a2, b2)
        elif name == "M":
            q = args[0]
            p_raw = sim.probs(q)
            s = int(frame[q][0]) if protocol == "frame" else 0
            p_corr = np.array([p_raw[(k - s) % 3] for k in range(3)])
            k_corr = sample_cdf(p_corr, draws[i])
            sim.project(q, (k_corr - s) % 3)
            record.append(k_corr)
            if q in ANCILLAS:
                round_buf.append(k_corr)
            if i in ends:
                synd = tuple(round_buf)
                round_buf = []
                corr = lookup.get(synd)
                if corr is not None:
                    if protocol == "physical":
                        for qq in DATA:
                            sim.weyl(qq, *corr[qq])
                    else:
                        frame = (frame + corr) % 3
        elif name == "RESET":
            q = args[0]
            p_raw = sim.probs(q)
            s = int(frame[q][0]) if protocol == "frame" else 0
            p_corr = np.array([p_raw[(k - s) % 3] for k in range(3)])
            k_corr = sample_cdf(p_corr, draws[i])
            k_raw = (k_corr - s) % 3
            sim.project(q, k_raw)
            sim.weyl(q, -k_raw, 0)
            frame[q] = (0, 0)
        else:
            raise KeyError(name)
    if not exact_logical:
        logical = (record[-2] + record[-1]) % 3
        return record, logical, None
    # exact distribution of corrected x0 + x3 after the remaining ops
    for name, args, params in ops[term_meas[0]:]:
        if name != "M":
            raise RuntimeError("unexpected op after terminal measurement")
    s0 = int(frame[0][0]) if protocol == "frame" else 0
    s3 = int(frame[3][0]) if protocol == "frame" else 0
    pj = sim.pair_marginal(0, 3)                   # raw (x0, x3)
    dist = np.zeros(3)
    for r0 in range(3):
        for r3 in range(3):
            dist[(r0 + s0 + r3 + s3) % 3] += pj[r0, r3]
    return record, None, dist / dist.sum()


# ----------------------------------------------------------------------------

def check_gates():
    print("Frame-update rule (conjugation U W U^dag), Weyl vectors (a,b) of X^a Z^b:")
    for name in ONE_Q:
        ix, iz = SYMP[name]
        print(f"  {name:6s}: X -> {ix}, Z -> {iz}")
    for name in TWO_Q:
        print(f"  {name:8s}: X_c -> {SYMP[name][0]}, Z_c -> {SYMP[name][1]}, "
              f"X_t -> {SYMP[name][2]}, Z_t -> {SYMP[name][3]}")


def check_noiseless(path):
    ops = parse_chp(path)
    rng = np.random.default_rng(0)
    draws = draw_noise(ops, rng, p_base=0.0)
    sim = Dense()
    det = []
    for i, (name, args, params) in enumerate(ops):
        if name in ONE_Q:
            sim.apply_named1(args[0], name)
        elif name in TWO_Q:
            sim.apply_named2(args[0], args[1], name)
        elif name in ("M", "RESET"):
            p = sim.probs(args[0])
            if args[0] in ANCILLAS:
                det.append(np.round(p, 12).tolist())
            k = sample_cdf(p, draws[i])
            sim.project(args[0], k)
            if name == "RESET":
                sim.weyl(args[0], -k, 0)
    pj = sim.pair_marginal(0, 3)
    dist = np.zeros(3)
    for r0 in range(3):
        for r3 in range(3):
            dist[(r0 + r3) % 3] += pj[r0, r3]
    anc_ok = all(abs(p[0] - 1) < 1e-9 for p in det)
    print(f"  noiseless {path.split('LRB/')[-1]}: ancilla/reset outcomes all "
          f"deterministic 0: {anc_ok}; P(x0+x3=k) = {np.round(dist, 12)}")
    return anc_ok and abs(dist[0] - 1) < 1e-9


def run_pauli(paths, shots, seed, p_base):
    total, mismatches, nontrivial = 0, 0, 0
    fid_phys = []
    for path in paths:
        ops = parse_chp(path)
        lookup = build_lookup(ops)
        rng = np.random.default_rng(seed)
        for _ in range(shots):
            draws = draw_noise(ops, rng, p_base)
            rec_p, log_p, _ = run_shot(ops, draws, lookup, "physical")
            rec_f, log_f, _ = run_shot(ops, draws, lookup, "frame")
            total += 1
            if rec_p != rec_f or log_p != log_f:
                mismatches += 1
            if any(rec_p[:-2]):
                nontrivial += 1
            fid_phys.append(W ** log_p)
    F = abs(np.mean(fid_phys))
    print(f"  Pauli mode: {total} shots over {len(paths)} circuits, p={p_base}: "
          f"{nontrivial} shots with a nontrivial syndrome, "
          f"{mismatches} shot(s) whose corrected record differs")
    print(f"  logical fidelity |E w^k| with the lookup decoder (both protocols): "
          f"{F:.4f}")
    return mismatches


def run_coherent(paths, shots, seed, theta):
    diffs, fp, ff = [], [], []
    for path in paths:
        ops = parse_chp(path)
        lookup = build_lookup(ops)
        rng = np.random.default_rng(seed + 1)
        for _ in range(shots):
            draws = draw_noise(ops, rng, p_base=None)
            _, _, dp = run_shot(ops, draws, lookup, "physical",
                                coherent_theta=theta, exact_logical=True)
            _, _, df = run_shot(ops, draws, lookup, "frame",
                                coherent_theta=theta, exact_logical=True)
            cp = np.dot(dp, W ** np.arange(3)).real
            cf = np.dot(df, W ** np.arange(3)).real
            fp.append(cp)
            ff.append(cf)
            diffs.append(cp - cf)
    fp, ff, diffs = map(np.asarray, (fp, ff, diffs))
    n = len(diffs)
    print(f"  coherent mode: theta={theta}, {n} shots: "
          f"Re E[w^k] physical = {fp.mean():.4f} +- {fp.std(ddof=1)/np.sqrt(n):.4f}, "
          f"frame = {ff.mean():.4f} +- {ff.std(ddof=1)/np.sqrt(n):.4f}")
    print(f"  paired difference physical - frame = {diffs.mean():.4f} +- "
          f"{diffs.std(ddof=1)/np.sqrt(n):.4f} "
          f"(fraction of shots with |diff|>1e-9: {np.mean(np.abs(diffs) > 1e-9):.3f})")
    return diffs.mean(), diffs.std(ddof=1) / np.sqrt(n)


def _chan_unitary(u):
    return lambda r: u @ r @ u.conj().T


def _chan_kraus(ks):
    return lambda r: sum(k @ r @ k.conj().T for k in ks)


def _weyl_twirl(chan):
    ws = [weyl(a, b) for a in range(3) for b in range(3)]
    return lambda r: sum(w.conj().T @ chan(w @ r @ w.conj().T) @ w
                         for w in ws) / 9.0


def _zdist(r):
    return np.real(np.diag(r)).copy()


def minimal_examples(theta=0.15, gamma=0.05):
    """Exact one-qutrit examples: where frame == physical and where not.

    Sequence: a Pauli error E on the input, noise N, the correction C = E^-1
    (applied physically, or kept in the frame), noise N, readout.  With the
    frame, the second N acts on the uncorrected state and the readout outcome
    is relabelled by the X power of the frame propagated to the readout.
    """
    print("Minimal exact one-qutrit examples, P(corrected outcome = 0,1,2):")
    plus = np.ones(3) / np.sqrt(3)
    rho_plus = np.outer(plus, plus.conj())
    zero = np.zeros((3, 3), complex)
    zero[0, 0] = 1
    hinv = H1.conj().T
    v = np.diag([np.exp(1j * theta * j) for j in range(3)])
    rng = np.random.default_rng(7)
    g = rng.normal(size=(3, 3)) + 1j * rng.normal(size=(3, 3))
    g = (g + g.conj().T) / 2
    evals, evecs = np.linalg.eigh(g)
    u_gen = evecs @ np.diag(np.exp(-1j * theta * evals)) @ evecs.conj().T
    k0 = np.diag([1, np.sqrt(1 - gamma), np.sqrt(1 - 2 * gamma)]).astype(complex)
    k1 = np.zeros((3, 3), complex)
    k1[0, 1] = np.sqrt(gamma)
    k2 = np.zeros((3, 3), complex)
    k2[1, 2] = np.sqrt(2 * gamma)
    amp = _chan_kraus([k0, k1, k2])
    xinv, zinv = X1.conj().T, Z1.conj().T
    cases = [
        ("Z-rotation V,         E=X^-1, C=X", _chan_unitary(v), xinv, rho_plus, hinv),
        ("Weyl-twirled V,       E=X^-1, C=X", _weyl_twirl(_chan_unitary(v)), xinv,
         rho_plus, hinv),
        ("Z-rotation V,         E=Z^-1, C=Z", _chan_unitary(v), zinv, rho_plus, hinv),
        ("generic unitary U,    E=X^-1, C=X", _chan_unitary(u_gen), xinv, rho_plus,
         hinv),
        ("Weyl-twirled U,       E=X^-1, C=X", _weyl_twirl(_chan_unitary(u_gen)), xinv,
         rho_plus, hinv),
        ("amplitude damping,    E=X,    C=X^-1", amp, X1, zero, I3),
        ("Weyl-twirled damping, E=X,    C=X^-1", _weyl_twirl(amp), X1, zero, I3),
    ]
    for label, chan, e, rho0, basis in cases:
        c = e.conj().T
        r1 = chan(e @ rho0 @ e.conj().T)
        r_phys = chan(c @ r1 @ c.conj().T)
        r_frame = chan(r1)
        shift = identify_weyl(basis @ c @ basis.conj().T, 1)[0]
        p_phys = _zdist(basis @ r_phys @ basis.conj().T)
        p_raw = _zdist(basis @ r_frame @ basis.conj().T)
        p_frame = np.array([p_raw[(k - shift) % 3] for k in range(3)])
        print(f"  {label:37s}: physical {np.round(p_phys, 5)}  "
              f"frame {np.round(p_frame, 5)}  max|diff| "
              f"{np.max(np.abs(p_phys - p_frame)):.2e}")


def main():
    global SYMP
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="all",
                    choices=["all", "gates", "noiseless", "pauli", "coherent",
                             "minimal"])
    ap.add_argument("--p-idx", default="8", help="baseline probability index "
                    "used for the circuit files (8 -> p=0.01)")
    ap.add_argument("--cliffs", default="0,1,2")
    ap.add_argument("--depth-idx", default="1", help="1 -> m=2 in depths.txt")
    ap.add_argument("--shots", type=int, default=200)
    ap.add_argument("--p", type=float, default=0.02,
                    help="rescaled SI1000 p for the Pauli mode")
    ap.add_argument("--theta", type=float, default=0.15)
    ap.add_argument("--coh-shots", type=int, default=300)
    ap.add_argument("--seed", type=int, default=12345)
    args = ap.parse_args()
    t0 = time.time()
    SYMP = symplectic_table()
    paths = [f"{RUN}/experiments/LRB/{c}/{args.p_idx}/{args.depth_idx}.chp"
             for c in args.cliffs.split(",")]
    if args.mode in ("all", "gates"):
        check_gates()
    if args.mode in ("all", "minimal"):
        minimal_examples()
    if args.mode in ("all", "noiseless"):
        for p in paths:
            check_noiseless(p)
        check_noiseless(f"{RUN}/experiments/LRB/0/{args.p_idx}/0.chp")
    if args.mode in ("all", "pauli"):
        run_pauli(paths, args.shots, args.seed, args.p)
    if args.mode in ("all", "coherent"):
        run_coherent(paths, args.coh_shots // len(paths), args.seed, args.theta)
    print(f"  elapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
