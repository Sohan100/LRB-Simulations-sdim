"""Check the general-q catalogue against production and report token counts.

1. q = 2, 3 production-mode catalogues equal the production tuples
   (same words, same order).
2. For q = 2, 3, 5, 7 (both compile modes): the catalogue has q^3 (q^2 - 1)
   distinct Cliffords (checked on sdim unitaries modulo phase) and every
   inverse word inverts its word.
3. Mean tokens per Clifford, the quantity that sets physical RB.
Writes results/catalogue_summary.csv.
"""
import csv
import sys
import numpy as np
import qslib
from lrb import code_definitions as defns

ok = True
prod = {2: defns.single_qubit_cliffords_exhaustive_strings,
        3: defns.single_qutrit_cliffords_exhaustive_strings}
for q, ref in prod.items():
    mine = list(qslib.catalogue(q, "production"))
    same = mine == list(ref)
    print(f"q={q}: production catalogue identical to repo tuple: {same}")
    ok &= same

def unitary(word, u):
    m = np.eye(u["I"].shape[0], dtype=complex)
    for tok in qslib.gate_sequence(word):      # execution order
        m = u[tok] @ m
    return m

def phase_key(m):
    flat = m.reshape(-1)
    i = int(np.argmax(np.abs(flat) > 1e-9))
    v = flat / (flat[i] / abs(flat[i]))
    return tuple(np.round(v.real, 6)) + tuple(np.round(v.imag, 6))

rows = []
for q in (2, 3, 5, 7):
    u = qslib.token_unitaries(q)
    for mode in ("production", "inverse"):
        cat = qslib.catalogue(q, mode)
        keys = set()
        inv_ok = True
        for w in cat:
            m = unitary(w, u)
            keys.add(phase_key(m))
            mi = np.eye(q, dtype=complex)
            for tok in qslib.inverse_sequence(w):
                mi = u[tok] @ mi
            prod_m = mi @ m
            inv_ok &= abs(abs(np.trace(prod_m)) - q) < 1e-6
        n_expected = q ** 3 * (q * q - 1)
        lens = np.array([len(w) for w in cat])
        sym = qslib.symplectic_words(q, "HS" if mode == "production" else "HShs")
        sym_len = np.array([0 if w == "I" else len(w) for w in sym])
        row = dict(q=q, compile=mode, n=len(cat), distinct=len(keys),
                   expected=n_expected, inverse_ok=inv_ok,
                   mean_tokens=lens.mean(), max_tokens=lens.max(),
                   mean_symplectic_tokens=sym_len.mean(),
                   max_symplectic_tokens=sym_len.max(),
                   mean_pauli_tokens=lens.mean() - sym_len.mean())
        rows.append(row)
        ok &= (len(keys) == n_expected == len(cat)) and inv_ok
        print(row)
with open("results/catalogue_summary.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
