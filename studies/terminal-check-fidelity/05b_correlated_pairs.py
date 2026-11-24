"""Which pairs of data qudits turn a correlated two-qudit error into an
accepted logical error at first order?

For every pair (a, b) a two-qudit depolarizing error of probability xt is
injected after every noisy Clifford, and the exact logical infidelity of one
mirror pair between perfect preparation and perfect check is differentiated
with respect to xt at p = 0 (so nothing else contributes). A non-zero slope
means single correlated faults on that pair are accepted logical errors.
Writes results/correlated_pairs.csv.
"""
from __future__ import annotations

import csv
import itertools
import os

import tclib as T
import tcsteps as S


def main():
    rows = []
    xt = 1e-6
    for code in ("qubit", "qutrit"):
        for a, b in itertools.combinations(range(5), 2):
            sm = S.StepModel(code, "si1000", crosstalk=((a, b),))
            slope = sm.ground_truth(1e-12, xt)["pair_stat"] / xt
            rows.append({"code": code, "pair": f"{a}-{b}", "d_eps_pair_d_xt": slope})
            print(code, f"{a}-{b}", f"{slope:.4f}", flush=True)
    with open(os.path.join(T.HERE, "results", "correlated_pairs.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
