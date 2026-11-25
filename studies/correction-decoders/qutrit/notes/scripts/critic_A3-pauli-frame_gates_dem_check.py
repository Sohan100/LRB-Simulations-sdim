#!/usr/bin/env python3
"""Critic of A3: independent check of the qutrit frame-update table against sdim.

What it does (read-only on the baseline run folder):
  1. Builds the one- and two-qutrit unitaries from sdim's own `sdim.unitary`
     generators (X, Z, H, P, CNOT) and derives CZ and SWAP from the gate
     sequences that sdim's *tableau* uses (`sdim/tableau/tableau_gates.py`:
     CZ = H_inv(t); CNOT; H(t), and the two SWAP decompositions).
  2. Extracts the symplectic action U W(v) U^dag = phase * W(S v) of every gate
     and compares it with the table claimed in A3 section 1.1.
  3. Checks whether sdim's tableau SWAP decompositions equal the standard SWAP
     exactly, up to a global phase, or only up to a Weyl factor.
  4. Propagates every unit Weyl component of every noise location of real
     baseline circuits with an independent linear frame propagator (written
     here from the table, not copied from A3) and compares the detector and
     logical responses with the repository's compiled DEM
     (`src/lrb/correction/compact_dem.py`, which calls sdim's frame simulator).

Usage: module load python/3.11; python critic_A3-pauli-frame_gates_dem_check.py
"""

from __future__ import annotations

import itertools
import os
import sys
import time

import numpy as np

WT = "/pscratch/sd/s/sohan100/worktrees/LRB-Simulations-sdim-2bc490701489f8e7/correction-decoders-qutrit-2c455b"
sys.path.insert(0, os.path.join(WT, "src"))
RUN = ("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
       "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")

from sdim import unitary as su  # noqa: E402

D = 3
w = np.exp(2j * np.pi / 3)
X = su.generate_x_matrix(D)
Z = su.generate_z_matrix(D)
H = su.generate_h_matrix(D)
P = su.generate_p_matrix(D)
CNOT = su.generate_cnot_matrix(D)       # basis index 3*c + t
I3 = np.eye(3)


def kron(*ms):
    out = np.array([[1.0 + 0j]])
    for m in ms:
        out = np.kron(out, m)
    return out


def weyl1(a, b):
    return np.linalg.matrix_power(X, a % 3) @ np.linalg.matrix_power(Z, b % 3)


def weyl(v):
    return kron(*[weyl1(v[2 * q], v[2 * q + 1]) for q in range(len(v) // 2)])


def identify(m, nq):
    for v in itertools.product(range(3), repeat=2 * nq):
        wv = weyl(v)
        k = np.argmax(np.abs(wv.flatten()))
        c = m.flatten()[k] / wv.flatten()[k]
        if abs(abs(c) - 1) < 1e-9 and np.allclose(m, c * wv, atol=1e-9):
            return tuple(int(x) for x in v), c
    return None, None


def on(u1, q, nq=2):
    ops = [I3] * nq
    ops[q] = u1
    return kron(*ops)


def cnot_ct(c, t, nq=2):
    """CNOT with control c, target t on nq qutrits (|i,j> -> |i, i+j>)."""
    dim = 3 ** nq
    u = np.zeros((dim, dim), complex)
    for idx in range(dim):
        dig = [(idx // 3 ** (nq - 1 - q)) % 3 for q in range(nq)]
        new = list(dig)
        new[t] = (dig[t] + dig[c]) % 3
        j = sum(new[q] * 3 ** (nq - 1 - q) for q in range(nq))
        u[j, idx] = 1
    return u


def seq(ops):
    """Time-ordered list of unitaries -> product (last op leftmost)."""
    u = np.eye(ops[0].shape[0], dtype=complex)
    for o in ops:
        u = o @ u
    return u


def main():
    t0 = time.time()
    Hd = H.conj().T
    # sanity: sdim's CNOT is |i,j> -> |i,i+j>
    assert np.allclose(CNOT, cnot_ct(0, 1)), "sdim CNOT convention differs"
    CNOT_INV = CNOT.conj().T
    # CZ from sdim's tableau sequence: hadamard_inv(t); cnot(c,t); hadamard(t)
    CZ_seq = seq([on(Hd, 1), CNOT, on(H, 1)])
    CZ_std = np.diag([w ** (i * j) for i in range(3) for j in range(3)])
    print(f"CZ (tableau sequence) == diag(w^ij): {np.allclose(CZ_seq, CZ_std)}")
    CZi_seq = seq([on(Hd, 1), CNOT_INV, on(H, 1)])
    print(f"CZ_INV (tableau sequence) == diag(w^-ij): "
          f"{np.allclose(CZi_seq, CZ_std.conj())}")
    SWAP_std = np.zeros((9, 9), complex)
    for i in range(3):
        for j in range(3):
            SWAP_std[3 * j + i, 3 * i + j] = 1
    # prime-Tableau SWAP branch: cnot(q,t) H(q) H(t) cnot(q,t) H(q) H(t) cnot(q,t) H(t) H(t)
    sw_prime = seq([CNOT, on(H, 0), on(H, 1), CNOT, on(H, 0), on(H, 1), CNOT,
                    on(H, 1), on(H, 1)])
    # WeylTableau branch: cnot(q,t) cnot_inv(t,q) cnot(q,t) H(q) H(q)
    sw_weyl = seq([CNOT, cnot_ct(1, 0).conj().T, CNOT, on(H, 0), on(H, 0)])
    for name, u in (("prime-Tableau SWAP sequence", sw_prime),
                    ("WeylTableau SWAP sequence", sw_weyl)):
        k = np.argmax(np.abs(u.flatten()))
        c = u.flatten()[k] / SWAP_std.flatten()[k] if SWAP_std.flatten()[k] else np.nan
        exact = np.allclose(u, c * SWAP_std) if np.isfinite(c) else False
        v, ph = identify(SWAP_std.conj().T @ u, 2)
        print(f"{name}: equals standard SWAP up to global phase: {exact}; "
              f"SWAP^dag U = phase*W{v} (phase {np.round(ph, 6) if ph is not None else None}); "
              f"max|U - SWAP| = {np.max(np.abs(u - SWAP_std)):.2e}")

    # symplectic table, compared with the A3 claims
    def img1(u):
        return [identify(u @ g @ u.conj().T, 1)[0] for g in (X, Z)]

    def img2(u):
        gens = (kron(X, I3), kron(Z, I3), kron(I3, X), kron(I3, Z))
        return [identify(u @ g @ u.conj().T, 2)[0] for g in gens]

    def lin1(imgs, a, b):
        return tuple((a * np.array(imgs[0]) + b * np.array(imgs[1])) % 3)

    def lin2(imgs, v):
        out = np.zeros(4, int)
        for coef, im in zip(v, imgs):
            out = (out + coef * np.array(im)) % 3
        return tuple(int(x) for x in out)

    claims1 = {
        "H": (H, lambda a, b: (-b % 3, a % 3)),
        "H_INV": (Hd, lambda a, b: (b % 3, -a % 3)),
        "P": (P, lambda a, b: (a % 3, (a + b) % 3)),
        "P_INV": (P.conj().T, lambda a, b: (a % 3, (b - a) % 3)),
    }
    ok_all = True
    for name, (u, rule) in claims1.items():
        im = img1(u)
        ok = all(lin1(im, a, b) == rule(a, b) for a in range(3) for b in range(3))
        ok_all &= ok
        print(f"  claim {name:6s}: {ok}")
    claims2 = {
        "CNOT": (CNOT, lambda ac, bc, at, bt: (ac, bc - bt, at + ac, bt)),
        "CNOT_INV": (CNOT_INV, lambda ac, bc, at, bt: (ac, bc + bt, at - ac, bt)),
        "CZ": (CZ_seq, lambda ac, bc, at, bt: (ac, bc + at, at, bt + ac)),
        "CZ_INV": (CZi_seq, lambda ac, bc, at, bt: (ac, bc - at, at, bt - ac)),
        "SWAP(prime seq)": (sw_prime, lambda ac, bc, at, bt: (at, bt, ac, bc)),
    }
    for name, (u, rule) in claims2.items():
        im = img2(u)
        ok = all(lin2(im, v) == tuple(x % 3 for x in rule(*v))
                 for v in itertools.product(range(3), repeat=4))
        ok_all &= ok
        print(f"  claim {name:15s}: {ok}")
    # readout rule: H_INV maps (a,b) -> (b,-a): X-power after H_INV is b
    print(f"All A3 frame-table entries reproduced from sdim matrices: {ok_all}")

    # ---------------------------------------------------------------- DEM check
    from lrb.correction.compact_dem import read_run_circuit as read_circuit
    from lrb.correction.compact_dem import build_compact_detector_error_model

    RULE1 = {"H": lambda a, b: (-b, a), "H_INV": lambda a, b: (b, -a),
             "P": lambda a, b: (a, a + b), "P_INV": lambda a, b: (a, b - a)}

    def propagate_all(path):
        """Independent batched linear frame propagation of every unit component."""
        lines = [ln.split() for ln in open(path)]
        ops = []
        for t in lines:
            if not t or t[0] in ("#", "Randomly-generated", "d"):
                continue
            args = [int(x) for x in t[1:] if "=" not in x]
            params = dict(x.split("=", 1) for x in t[1:] if "=" in x)
            ops.append((t[0], args, {k: v.strip('"') for k, v in params.items()}))
        # one batch member per (noise op index, component)
        members = []
        for i, (n, a, pr) in enumerate(ops):
            if n == "N1" and float(pr["prob"]) > 0:
                members += [(i, c) for c in range(2)]
            elif n == "N2" and float(pr["prob"]) > 0:
                members += [(i, c) for c in range(4)]
        B = len(members)
        fa = np.zeros((B, 9), int)
        fb = np.zeros((B, 9), int)
        by_op = {}
        for k, (i, c) in enumerate(members):
            by_op.setdefault(i, []).append((k, c))
        meas = []
        for i, (n, a, pr) in enumerate(ops):
            if n in RULE1:
                q = a[0]
                na, nb = RULE1[n](fa[:, q], fb[:, q])
                fa[:, q], fb[:, q] = na % 3, nb % 3
            elif n in ("X", "X_INV", "Z", "Z_INV", "I"):
                pass
            elif n in ("CNOT", "CNOT_INV", "CZ", "CZ_INV", "SWAP"):
                c, t = a
                ac, bc, at, bt = fa[:, c].copy(), fb[:, c].copy(), fa[:, t].copy(), fb[:, t].copy()
                if n == "CNOT":
                    fb[:, c] = bc - bt; fa[:, t] = at + ac
                elif n == "CNOT_INV":
                    fb[:, c] = bc + bt; fa[:, t] = at - ac
                elif n == "CZ":
                    fb[:, c] = bc + at; fb[:, t] = bt + ac
                elif n == "CZ_INV":
                    fb[:, c] = bc - at; fb[:, t] = bt - ac
                else:
                    fa[:, c], fb[:, c], fa[:, t], fb[:, t] = at, bt, ac, bc
                fa %= 3; fb %= 3
            elif n == "M":
                meas.append(fa[:, a[0]].copy())
            elif n == "RESET":
                fa[:, a[0]] = 0; fb[:, a[0]] = 0
            elif n in ("N1", "N2"):
                for k, c in by_op.get(i, []):
                    q = a[c // 2]
                    if c % 2 == 0:
                        fa[k, q] = (fa[k, q] + 1) % 3
                    else:
                        fb[k, q] = (fb[k, q] + 1) % 3
            elif n in ("DETECTOR", "LOGICAL_OBSERVABLE", "TICK"):
                pass
            else:
                raise KeyError(n)
        meas = np.array(meas).T          # (B, n_meas)
        det = meas[:, :-2] % 3
        log = (meas[:, -2] + meas[:, -1]) % 3
        return ops, members, det, log

    def compare(path):
        ops, members, det, log = propagate_all(path)
        circ = read_circuit(path)
        model = build_compact_detector_error_model(circ)
        # map DEM locations (operation_index in sdim circuit) to our op order:
        # both enumerate N1/N2 in file order, so match by rank among noise ops
        noise_ops = [i for i, (n, _, pr) in enumerate(ops)
                     if n in ("N1", "N2") and float(pr["prob"]) > 0]
        rank = {i: r for r, i in enumerate(noise_ops)}
        dem_rank = {}
        for r, loc in enumerate(model.locations):
            dem_rank[r] = loc
        mism = 0
        n_checked = 0
        for k, (i, c) in enumerate(members):
            loc = dem_rank[rank[i]]
            terms = loc.component_terms[c]
            dv = np.zeros(det.shape[1], int)
            for idx, val in terms.detector:
                dv[idx] = val
            lv = 0
            for idx, val in terms.logical:
                lv = val
            if not (np.array_equal(dv, det[k]) and lv == log[k]):
                mism += 1
            n_checked += 1
        return n_checked, mism, model.num_detectors

    for rel in ("experiments/LRB/0/8/1.chp", "experiments/LRB/1/8/1.chp",
                "experiments/LRB/2/8/1.chp", "experiments/LRB/0/8/3.chp",
                "experiments/LRB/7/8/3.chp"):
        n, mm, nd = compare(f"{RUN}/{rel}")
        print(f"DEM vs independent frame propagation, {rel}: {n} unit components, "
              f"{nd} detectors, mismatches = {mm}")
    print(f"elapsed {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
