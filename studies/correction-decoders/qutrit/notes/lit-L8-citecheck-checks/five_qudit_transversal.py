"""Independent check: which uniform transversal single-qudit Cliffords (mod Paulis, i.e. SL(2,q))
preserve the stabilizer space of the 5-qudit code with generators = cyclic shifts of I Z X X^-1 Z^-1
(Keppens et al. 2025 convention). Symplectic vectors: X^a Z^b -> (a, b)."""
import itertools, os
os.environ.setdefault('OMP_NUM_THREADS', '1')
import numpy as np

def rank_mod(A, q):
    A = A.copy() % q
    r = 0
    rows, cols = A.shape
    for c in range(cols):
        piv = None
        for i in range(r, rows):
            if A[i, c] % q:
                piv = i; break
        if piv is None: continue
        A[[r, piv]] = A[[piv, r]]
        inv = pow(int(A[r, c]), -1, q)
        A[r] = (A[r] * inv) % q
        for i in range(rows):
            if i != r and A[i, c]:
                A[i] = (A[i] - A[i, c] * A[r]) % q
        r += 1
    return r

def code(q):
    # single-qudit Paulis as (a,b): I=(0,0), Z=(0,1), X=(1,0), X^-1=(-1,0), Z^-1=(0,-1)
    g = [(0, 0), (0, 1), (1, 0), (q - 1, 0), (0, q - 1)]
    gens = []
    for s in range(4):
        sh = g[-s:] + g[:-s] if s else g
        gens.append([a for a, b in sh] + [b for a, b in sh])
    return np.array(gens, dtype=int)

def apply_local(G, M, q, n=5):
    out = G.copy()
    for i in range(n):
        a, b = G[:, i], G[:, n + i]
        # columns of M are images of X and Z: X^a Z^b -> a*M[:,0] + b*M[:,1]
        out[:, i] = (M[0, 0] * a + M[0, 1] * b) % q
        out[:, n + i] = (M[1, 0] * a + M[1, 1] * b) % q
    return out

def order(M, q):
    P = np.eye(2, dtype=int); k = 0
    while True:
        P = (P @ M) % q; k += 1
        if (P == np.eye(2, dtype=int)).all(): return k

import sys
QS = tuple(int(a) for a in sys.argv[1:]) or (2, 3)
for q in QS:
    G = code(q)
    r0 = rank_mod(G, q)
    found = []
    for a, b, c, d in itertools.product(range(q), repeat=4):
        if (a * d - b * c) % q != 1: continue
        M = np.array([[a, b], [c, d]])
        G2 = apply_local(G, M, q)
        if rank_mod(np.vstack([G, G2]), q) == r0:
            found.append(((a, b, c, d), order(M, q)))
    print('q=%d rank=%d  preserving SL(2,q) elements (a,b,c,d; order):' % (q, r0), found)
    # specific T = S F with S: X->XZ, Z->Z and F: X->Z, Z->X^-1 (apply F first): X->Z, Z->(XZ)^-1
    S = np.array([[1, 0], [1, 1]]); F = np.array([[0, q - 1], [1, 0]])
    for name, T in (('S*F (F first)', (S @ F) % q), ('F*S (S first)', (F @ S) % q)):
        G2 = apply_local(G, T, q)
        ok = rank_mod(np.vstack([G, G2]), q) == r0
        print('   %s = %s, order %d, preserves stabilizer space: %s' % (name, T.flatten().tolist(), order(T, q), ok))
