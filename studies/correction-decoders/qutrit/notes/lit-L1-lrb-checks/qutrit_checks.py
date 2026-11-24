"""Small numerical checks for the LRB literature notes (single qutrit, d = 3).

1. Generate the single-qutrit Clifford group from H (Fourier) and S = diag(1,1,w)
   modulo global phase, count elements, compute frame potentials
   FP_t = mean_g |Tr g|^(2t) for t = 1, 2, 3 (Haar values 1, 2, 6 for d = 3).
2. Mirrored ("Clifford-by-Clifford inverse in reverse order") sequences versus
   standard RB (single recovery Clifford) under gate-independent Weyl noise:
   compare the Monte Carlo average of <Z> = sum_k w^k P(k) with the frame-argument
   predictions.
3. Real part vs modulus of <Z> per sequence for inversion-symmetric and
   asymmetric Weyl channels.
"""
import itertools
import numpy as np

rng = np.random.default_rng(12345)
d = 3
w = np.exp(2j * np.pi / d)
F = np.array([[w ** (j * k) for k in range(d)] for j in range(d)]) / np.sqrt(d)
S = np.diag([1, 1, w])
X = np.roll(np.eye(d), 1, axis=0)          # X|j> = |j+1>
Z = np.diag([w ** j for j in range(d)])


def canon(U):
    flat = U.flatten()
    idx = np.argmax(np.abs(flat) > 1e-9)
    ph = flat[idx] / abs(flat[idx])
    V = U / ph
    return tuple(np.round(V.flatten(), 8).tolist())


def closure(gens):
    elems = {canon(np.eye(d)): np.eye(d, dtype=complex)}
    frontier = [np.eye(d, dtype=complex)]
    while frontier:
        new = []
        for A in frontier:
            for G in gens:
                B = G @ A
                key = canon(B)
                if key not in elems:
                    elems[key] = B
                    new.append(B)
        frontier = new
    return list(elems.values())


group = closure([F, S])
print("single-qutrit Clifford group order (mod phase):", len(group),
      " expected d^3(d^2-1) =", d ** 3 * (d ** 2 - 1))
# check X and Z are in the group
keys = {canon(g) for g in group}
print("X in group:", canon(X) in keys, " Z in group:", canon(Z) in keys)
tr = np.array([abs(np.trace(g)) for g in group])
for t in (1, 2, 3):
    print(f"frame potential t={t}: {np.mean(tr ** (2 * t)):.6f}")

# Weyl operators W(a,b) = X^a Z^b
W = {(a, b): np.linalg.matrix_power(X, a) @ np.linalg.matrix_power(Z, b)
     for a in range(d) for b in range(d)}


def weyl_channel(q):
    """q: dict (a,b)->prob. returns function rho -> sum q W rho W^dag"""
    items = [(W[v], p) for v, p in q.items() if p > 0]

    def apply(rho):
        out = np.zeros_like(rho)
        for Wv, p in items:
            out += p * (Wv @ rho @ Wv.conj().T)
        return out
    return apply


def polarization(q):
    return (d * d * q.get((0, 0), 0.0) - 1) / (d * d - 1)


def compose(q1, q2):
    out = {}
    for (a1, b1), p1 in q1.items():
        for (a2, b2), p2 in q2.items():
            v = ((a1 + a2) % d, (b1 + b2) % d)
            out[v] = out.get(v, 0.0) + p1 * p2
    return out


def lamZ(q):
    # <Z> multiplier for a lab-frame Weyl error X^a Z^b: W^dag Z W = w^{?} Z
    tot = 0
    for (a, b), p in q.items():
        Wv = W[(a, b)]
        M = Wv.conj().T @ Z @ Wv
        tot += p * (M[0, 0] / Z[0, 0])
    return tot


def run(qf, qb, m, nseq, mirrored=True):
    Ef, Eb = weyl_channel(qf), weyl_channel(qb)
    vals = np.empty(nseq, dtype=complex)
    rho0 = np.zeros((d, d), dtype=complex)
    rho0[0, 0] = 1
    for s in range(nseq):
        idx = rng.integers(len(group), size=m)
        rho = rho0.copy()
        if mirrored:
            for i in idx:
                C = group[i]
                rho = Ef(C @ rho @ C.conj().T)
            for i in idx[::-1]:
                C = group[i]
                rho = Eb(C.conj().T @ rho @ C)
        else:
            Ctot = np.eye(d, dtype=complex)
            for i in idx:
                C = group[i]
                rho = Ef(C @ rho @ C.conj().T)
                Ctot = C @ Ctot
            R = Ctot.conj().T
            rho = Eb(R @ rho @ R.conj().T)
        vals[s] = np.trace(Z @ rho)
    return vals


eps = 0.2
q_dep = {(0, 0): 1 - eps}
for v in itertools.product(range(d), repeat=2):
    if v != (0, 0):
        q_dep[v] = eps / 8
q_zsym = {(0, 0): 1 - eps, (0, 1): eps / 2, (0, 2): eps / 2}   # sparse, inversion-symmetric
q_zasym = {(0, 0): 1 - eps, (0, 1): eps}                       # sparse, not inversion-symmetric

m, nseq = 8, 6000
for name, q in (("depolarizing", q_dep), ("Z-dephasing (symmetric)", q_zsym)):
    vals = run(q, q, m, nseq, mirrored=True)
    g = polarization(q)
    gpair = polarization(compose(q, q))
    pred_frame = lamZ(q) * g * gpair ** (m - 1)
    pred_naive = lamZ(q) * g * (g * g) ** (m - 1)
    se = vals.real.std() / np.sqrt(nseq)
    print(f"[mirrored, m={m}] {name}: MC mean Re<Z> = {vals.real.mean():.5f} +- {se:.5f}; "
          f"frame prediction = {pred_frame.real:.5f}; product-of-polarizations = {pred_naive.real:.5f}; "
          f"gamma={g:.4f}, gamma(pair)={gpair:.4f}, gamma^2={g*g:.4f}")
    vals_s = run(q, q, m, nseq, mirrored=False)
    pred_std = lamZ(q) * g ** m
    print(f"[standard RB, m={m}] {name}: MC mean Re<Z> = {vals_s.real.mean():.5f} +- "
          f"{vals_s.real.std()/np.sqrt(nseq):.5f}; prediction lamZ*gamma^m = {pred_std.real:.5f}")

# 3. modulus vs real part
for name, q in (("symmetric Z-dephasing", q_zsym), ("asymmetric Z only", q_zasym)):
    vals = run(q, q, 4, 3000, mirrored=True)
    print(f"[m=4 mirrored] {name}: max |Im<Z>| per sequence = {np.abs(vals.imag).max():.2e}; "
          f"mean Re = {vals.real.mean():.5f}; mean |.| = {np.abs(vals).mean():.5f}")
