"""Re-sample the production const-0 circuits with sdim, keeping the full record.

The production tallies keep only the accepted shots by logical value and the
number of rejected shots, not which stabilizer fired. The mode fit needs the
full terminal record (s1, s2, l), so this script reads the production
``experiments/LRB_const0`` .chp files (read-only), samples each with sdim's
own DEM sampler, and histograms the d^3 outcomes. Nothing is written to the
run folders; counts go to results/resample_<code>_<noise>_p<idx>.npz.

It then applies the estimators with a one-standard-error bootstrap over the
30 sequences and compares them with the exact reference values.

Usage: python 06_production_resample.py [--shots 1000000] [--p-indices 4,6]
"""
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import tclib as T
import tcfit as F
import tcsteps as S
from tcref import reference

from sdim.circuit_io import read_circuit
from sdim.dem import DetectorErrorModel


def sample_one(args):
    path, d, shots, seed = args
    circ = read_circuit(path)
    has_noise = any(op.gate_name in ("N1", "N2") for op in circ.operations)
    hist = np.zeros(d ** 3, dtype=np.int64)
    if not has_noise:                      # depth 0: noiseless, every shot is outcome 0
        hist[0] = shots
        return hist
    dem = DetectorErrorModel.from_circuit(circ)
    sampler = dem.compile_sampler(seed=seed)
    left = shots
    while left > 0:
        n = min(left, 250_000)
        det, obs = sampler.sample(n)
        idx = (np.asarray(det[:, 0], np.int64) % d) * d * d + (np.asarray(det[:, 1], np.int64) % d) * d \
            + (np.asarray(obs[:, 0], np.int64) % d)
        hist += np.bincount(idx, minlength=d ** 3)
        left -= n
    return hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=1_000_000)
    ap.add_argument("--p-indices", default="4,6")
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    rows = []
    for key in T.RUNS:
        code, noise = key
        run = os.path.join(T.MAIN_DATA, T.RUNS[key])
        sm = S.StepModel(code, noise)
        d = sm.d
        F.modes(d)  # fills the mode cache
        depths = [int(x) for x in open(os.path.join(run, "depths.txt")).read().split(",")]
        probs = [float(x) for x in open(os.path.join(run, "probs.txt")).read().split(",")]
        for pi in [int(x) for x in a.p_indices.split(",")]:
            out = os.path.join(T.HERE, "results", f"resample_{code}_{noise}_p{pi}.npz")
            if os.path.exists(out):
                cnt = np.load(out)["counts"]
            else:
                jobs = [(os.path.join(run, "experiments", "LRB_const0", str(s), str(pi), f"{di}.chp"), d, a.shots,
                         1_000_003 * pi + 1009 * s + di)
                        for s in range(30) for di in range(len(depths))]
                with ProcessPoolExecutor(max_workers=a.workers) as ex:
                    res = list(ex.map(sample_one, jobs, chunksize=4))
                cnt = np.array(res).reshape(30, len(depths), d ** 3)
                np.savez_compressed(out, counts=cnt, depths=np.array(depths), p=probs[pi], shots=a.shots)
            p = probs[pi]
            gt = sm.ground_truth(p)
            ref = reference(code, noise, p)
            beta_true, fcov = ref["coincidence"], ref["f_cov"]

            def est(c):
                mb = F.mode_fit(depths, c, d, "block")
                mf = F.mode_fit(depths, c, d, "free")
                se = F.single_exponential(depths, c, d)
                return {"block_beta": mb["eps1"], "free_eps1": mf["eps1"], "free_alpha": mf["alpha"],
                        "discard1": mb["discard1"], "single_exp_r": se["r"]}

            point = est(cnt.sum(0))
            boot = F.bootstrap(est, cnt, n_boot=300, seed=pi)
            rows.append({
                "code": code, "noise": noise, "p": p, "shots_per_circuit": int(cnt[0, 1].sum()),
                "true_pair": gt["pair_stat"], "true_coincidence": beta_true, "f_cov": fcov,
                "true_discard_pair": gt["pair_stat_discard"],
                "block_beta": point["block_beta"], "block_beta_se": boot["block_beta"][1],
                "block_times_fcov": point["block_beta"] * fcov, "block_times_fcov_se": boot["block_beta"][1] * fcov,
                "free_eps1": point["free_eps1"], "free_eps1_se": boot["free_eps1"][1],
                "discard1": point["discard1"], "discard1_se": boot["discard1"][1],
                "single_exp_r": point["single_exp_r"], "single_exp_r_se": boot["single_exp_r"][1],
            })
            r = rows[-1]
            print(f"{code} {noise} p={p:g}: true pair {r['true_pair']:.3e} (coincidence {r['true_coincidence']:.3e}, F {fcov:.3f}) | "
                  f"block {r['block_beta']:.3e} +- {r['block_beta_se']:.1e} -> x F = {r['block_times_fcov']:.3e} +- {r['block_times_fcov_se']:.1e} | "
                  f"free {r['free_eps1']:.3e} +- {r['free_eps1_se']:.1e} | single-exp {r['single_exp_r']:.3e} +- {r['single_exp_r_se']:.1e} | "
                  f"discard {r['discard1']:.4f} +- {r['discard1_se']:.4f} (true {r['true_discard_pair']:.4f})", flush=True)
    path = os.path.join(T.HERE, "results", "production_resample.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path)


if __name__ == "__main__":
    main()
