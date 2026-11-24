"""Random-acceptance onset of LRB-D at large q, from exact DEMs.

A shot whose detectors are fully scrambled passes the D_m = 4(m+1) checks of a
depth-m circuit with probability q^-D_m and then carries a uniformly random
logical value.  Shots in which no fault is visible pass with probability
C_m(p) = prod_i [1 - pi_i(p) (1 - z_i)], z_i = Pr[an element of V_i has a zero
detector image] = q^-rank(det V_i).  Write Lambda_m = -ln C_m.  Then
    A_m(p) ~ C_m(p) + q^-D_m,
and the accepted set is dominated by random shots once Lambda_m > D_m ln q.
For the high-p fits (depths 0 and 1) this defines
    p_rand(q):  Lambda_1(p_rand) = 8 ln q,
above which the fitted LRB-D error jumps towards (q-1)/q.

For q = 2..31, SI1000 and SD6, both compiles: mean Lambda_0, Lambda_1 over
n random depth-1 sequences (depth 0 has no Clifford), the slopes
N_m = dLambda_m/dp at p -> 0, and p_rand.  Writes results/random_acceptance.csv.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from scipy.optimize import brentq

import qpairs
import qslib


def lam_terms(q, noise, mode, words, depth):
    from lrb.circuit_generator import LRBCircuitGenerator
    from lrb.lrb_simulation import NoiseModelUtils
    from sdim.dem import DetectorErrorModel

    cd = qslib.code_definition_for_words(q, words or ["I"], mode)
    gen = LRBCircuitGenerator(code_definition=cd, noise_model=noise)
    c = gen.generate_lrb_clifford_sequence([depth], clifford_strings=list(words))[0]
    p_ref = 1e-4
    gen.update_noise_param(c, p_ref)
    NoiseModelUtils.ensure_noise_params(c)
    dem = DetectorErrorModel.from_circuit(c, merge=False)
    D = dem.num_detectors
    coef, vis = [], []
    for m in dem.mechanisms:
        G = np.zeros((len(m.generators), D), dtype=np.int64)
        for r, g in enumerate(m.generators):
            for t, v in g.items():
                if int(t) < D:
                    G[r, int(t)] = int(v) % q
        rk = qpairs.rank_mod(G, q) if G.size else 0
        coef.append(m.probability / p_ref)
        vis.append(1.0 - float(q) ** (-rk))
    return np.array(coef), np.array(vis), D


def job(args):
    q, noise, mode, seed = args
    words = qslib.random_words(q, mode, 1, seed)
    c1, v1, D1 = lam_terms(q, noise, mode, words, 1)
    return q, noise, mode, seed, c1, v1, D1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="2,3,5,7,11,13,17,19,23,29,31")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    qs = [int(x) for x in a.q.split(",")]
    rows = []
    depth0 = {}
    for q in qs:
        for noise in ("si1000", "sd6"):
            c0, v0, D0 = lam_terms(q, noise, "production", [], 0)
            depth0[(q, noise)] = (c0, v0, D0)
    jobs = [(q, noise, mode, 1000 * q + s)
            for q in qs for noise in ("si1000", "sd6") for mode in ("production", "inverse")
            if not (q == 2 and mode == "inverse") for s in range(a.n)]
    res = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for q, noise, mode, seed, c1, v1, D1 in ex.map(job, jobs):
            res.setdefault((q, noise, mode), []).append((c1, v1, D1))
    for (q, noise, mode), lst in sorted(res.items()):
        c0, v0, D0 = depth0[(q, noise)]
        D1 = lst[0][2]

        def lam1(p):
            vals = [-np.sum(np.log1p(-np.minimum(c * p, 1.0) * v)) for c, v, _ in lst]
            # average acceptance over sequences, then back to an exponent
            return -np.log(np.mean(np.exp(-np.array(vals))))

        def lam0(p):
            return -np.sum(np.log1p(-np.minimum(c0 * p, 1.0) * v0))

        pmax = 0.999 / max(max(c.max() for c, _, _ in lst), c0.max())
        target = D1 * np.log(q)
        try:
            prand = brentq(lambda p: lam1(p) - target, 1e-6, pmax)
        except ValueError:
            prand = np.nan
        N1 = np.mean([np.sum(c * v) for c, v, _ in lst])
        N0 = float(np.sum(c0 * v0))
        rows.append(dict(q=q, noise=noise, compile=mode, nbar=qslib.mean_tokens_exact(q, mode),
                         D0=D0, D1=D1, N0=N0, N1=N1, p_rand=prand,
                         lam1_at_0p05=lam1(0.05) if 0.05 < pmax else np.nan,
                         A1_model_0p05=(np.exp(-lam1(0.05)) + q ** -float(D1)) if 0.05 < pmax else np.nan))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(f"results/random_acceptance{qslib.ROUND_SUFFIX}.csv", index=False)


if __name__ == "__main__":
    main()
