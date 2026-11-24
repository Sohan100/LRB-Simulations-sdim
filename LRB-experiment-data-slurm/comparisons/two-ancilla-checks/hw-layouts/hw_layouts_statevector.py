"""Exact statevector check: do the chosen check rounds leave a logical state alone?

A generic (random, non-stabilizer) logical qutrit state |psi> is prepared in the
[[5,1,2]]_3 code space, the ancillas start in |0>, and the noiseless rounds of
hw_layouts_check.SCHEMES are applied gate by gate on the full 9-qutrit state
(3^9 amplitudes). After every check we report the readout probabilities, the
fidelity of the data with |psi>, and <G1..G4> on the data. A per-gate trace of
the left-layout G2 relay shows where the stabilizers go during the relay, and
the naive split of G2 over two independent readouts is run as a contrast.

Conventions as in sdim 1.4.0: X|j> = |j+1>, Z|j> = w^j |j>, CNOT c t:
|c,t> -> |c,t+c>, CNOT_INV subtracts, H is the Fourier transform.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import hw_layouts_check as H  # noqa: E402

W = np.exp(2j * np.pi / 3)
X1 = np.roll(np.eye(3), 1, axis=0)
Z1 = np.diag([1, W, W ** 2])
F = np.array([[W ** (j * k) for j in range(3)] for k in range(3)]) / np.sqrt(3)
NW, ND = 9, 5


def pauli(kind, vec):
    base = X1 if kind == "X" else Z1
    out = np.array([[1.0 + 0j]])
    for e in vec:
        out = np.kron(out, np.linalg.matrix_power(base, e % 3))
    return out


G = {k: pauli(*H.STAB[k]) for k in ("G1", "G2", "G3", "G4")}
XL, ZL = pauli("X", (1, 0, 0, 1, 0)), pauli("Z", (1, 0, 1, 0, 0))   # X0 X3, Z0 Z2
P = np.eye(3 ** ND, dtype=complex)
for M in G.values():
    P = P @ (np.eye(3 ** ND) + M + M @ M) / 3


def embed_data(op):
    """op on the 5 data qutrits, identity on the ancillas, as a function on psi."""
    def act(psi):
        flat = psi.reshape(3 ** ND, -1)
        return (op @ flat).reshape(psi.shape)
    return act


def expect(psi, op):
    return np.vdot(psi, embed_data(op)(psi)).real


def gate(psi, name, *w):
    if name in ("CNOT", "CNOT_INV"):
        c, t = w
        s = 1 if name == "CNOT" else -1
        out = np.empty_like(psi)
        for v in range(3):
            idx = [slice(None)] * NW
            idx[c] = v
            out[tuple(idx)] = np.roll(psi[tuple(idx)], s * v, axis=t if t < c else t - 1)
        return out
    M = F if name == "H" else F.conj().T
    return np.moveaxis(np.tensordot(M, psi, axes=([1], [w[0]])), 0, w[0])


def run(psi, ops, rng, trace=None):
    """Apply ops; return final psi and a list of (tag, outcome, probabilities)."""
    last, log = {}, []
    for o in ops:
        if o[0] == "g":
            psi = gate(psi, *o[1:])
            if trace is not None:
                trace(o, psi)
        elif o[0] == "m":
            w = o[1]
            p = np.array([np.sum(np.abs(np.take(psi, v, axis=w)) ** 2) for v in range(3)])
            k = rng.choice(3, p=p / p.sum())
            keep = np.zeros(3)
            keep[k] = 1
            psi = psi * keep.reshape([3 if i == w else 1 for i in range(NW)])
            psi /= np.linalg.norm(psi)
            last[w] = k
            log.append((o[2], k, p))
        else:
            psi = np.roll(psi, -last.get(o[1], 0), axis=o[1])
    return psi, log


def data_state(psi):
    return psi[(slice(None),) * ND + (0,) * (NW - ND)].reshape(-1)


def report_check(psi, psi0, log):
    d = data_state(psi)
    fid = abs(np.vdot(psi0, d)) ** 2
    stabs = " ".join(f"<{k}>={np.vdot(d, M @ d).real:+.3f}" for k, M in G.items())
    reads = ", ".join(f"{t}: P(0)={p[0]:.6f} got {k}" for t, k, p in log)
    return f"fidelity {fid:.12f}  {stabs}  | {reads}"


def split_checks(ops):
    """Split a round into one chunk per check (a chunk ends after its resets)."""
    chunks, cur = [], []
    for i, o in enumerate(ops):
        cur.append(o)
        nxt = ops[i + 1] if i + 1 < len(ops) else None
        if o[0] == "r" and (nxt is None or nxt[0] != "r"):
            chunks.append(cur)
            cur = []
    return chunks


def main():
    rng = np.random.default_rng(20261116)
    evals, evecs = np.linalg.eigh(P)
    code = evecs[:, evals > 0.5]
    print(f"code space dimension {code.shape[1]}")
    names = ["one ancilla (reference)",
             "LEFT  G1,G2(a reads),G3,G3G4^-1 hybrid q2-q3",
             "RIGHT G1(a reads),G2,G3,G4 via q2-q3"]
    worst = {n: (1.0, 0.0) for n in names}
    for trial in range(5):
        coeff = rng.normal(size=3) + 1j * rng.normal(size=3)
        psi0 = code @ (coeff / np.linalg.norm(coeff))
        if trial == 0:
            print(f"random logical state: <X_L>={np.vdot(psi0, XL @ psi0):.3f} "
                  f"<Z_L>={np.vdot(psi0, ZL @ psi0):.3f}")
        for name in names:
            psi = np.zeros((3,) * NW, dtype=complex)
            psi[(slice(None),) * ND + (0,) * (NW - ND)] = psi0.reshape((3,) * ND)
            if trial == 0:
                print(f"\n{name}")
            for rnd in range(3):
                for chunk in split_checks(H.SCHEMES[name]):
                    psi, log = run(psi, chunk, rng)
                    d = data_state(psi)
                    fid = abs(np.vdot(psi0, d)) ** 2
                    pmin = min(p[0] for _, _, p in log)
                    worst[name] = (min(worst[name][0], fid), max(worst[name][1], 1 - pmin))
                    if trial == 0 and rnd == 0:
                        print("  " + report_check(psi, psi0, log))
            if trial == 0:
                d = data_state(psi)
                print(f"  after 3 rounds: fidelity {abs(np.vdot(psi0, d)) ** 2:.12f}")
    print("\nover 5 random logical states x 3 rounds:")
    for n, (f, p) in worst.items():
        print(f"  {n:46s} min fidelity {f:.12f}  max P(readout != 0) {p:.1e}")

    # per-gate trace of the left G2 relay
    coeff = rng.normal(size=3) + 1j * rng.normal(size=3)
    psi0 = code @ (coeff / np.linalg.norm(coeff))
    psi = np.zeros((3,) * NW, dtype=complex)
    psi[(slice(None),) * ND + (0,) * (NW - ND)] = psi0.reshape((3,) * ND)
    lab = {0: "q0", 1: "q1", 2: "q2", 3: "q3", 4: "q4", 5: "a", 6: "b"}

    def tr(o, s):
        a_zero = np.sum(np.abs(np.take(s, 0, axis=5)) ** 2)
        b_zero = np.sum(np.abs(np.take(s, 0, axis=6)) ** 2)
        st = " ".join(f"<{k}>={expect(s, M):+.3f}" for k, M in G.items())
        print(f"  {o[1]:8s} {lab[o[2]]}->{lab[o[3]]:3s} P(a=0)={a_zero:.3f} P(b=0)={b_zero:.3f}  "
              f"<data in code>={expect(s, P):.3f}  {st}")

    print("\nLEFT G2 relay, gate by gate (data stabilizers measured on the data alone):")
    psi, log = run(psi, H.L_G2_readA, rng, trace=tr)
    print("  " + report_check(psi, psi0, log))

    # contrast: G2 split into two independent readouts
    naive = H.z_direct(H.A, [(1, 1), (2, 1)], "a: z1+z2") + H.z_direct(H.B, [(4, -1)], "b: -z4")
    psi = np.zeros((3,) * NW, dtype=complex)
    psi[(slice(None),) * ND + (0,) * (NW - ND)] = psi0.reshape((3,) * ND)
    psi, log = run(psi, naive, rng)
    print("\nnaive split of G2 (a reads z1+z2, b reads -z4, add the readouts):")
    print("  " + report_check(psi, psi0, log))


if __name__ == "__main__":
    main()
