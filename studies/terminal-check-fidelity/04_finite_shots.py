"""Finite-shot, finite-sequence performance of the terminal-check estimators.

Each scenario fixes a code, noise model, p, closing protocol, SPAM, depth
grid, number of random sequences and shots per circuit. For each scenario
``--reps`` synthetic experiments are generated: random sequences, their exact
outcome distributions (tcsteps), multinomial shots. Every estimator is run on
every experiment and compared with the exact reference values of
tcref.py (block and direct estimates against the coincidence infidelity,
the free estimate against the true unit infidelity). For the first ``--boot-reps`` experiments the
one-standard-error sequence bootstrap of the block estimate is also computed,
to check that it reproduces the true scatter.

Writes results/finite_shots.csv. About 6 minutes on 16 login-node cores.
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

from tcref import design_depths, reference
PROD_DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
_SM = {}


def model(code, noise):
    if (code, noise) not in _SM:
        _SM[(code, noise)] = S.StepModel(code, noise)
    return _SM[(code, noise)]


def run_scenario(sc):
    sm = model(sc["code"], sc["noise"])
    d = sm.d
    F.modes(d)  # fills the mode cache
    p = sc["p"]
    gt = sm.ground_truth(p)
    mirror = sc["protocol"] == "mirror"
    true_unit = gt["pair_stat"] if mirror else gt["single_stat"]
    true_beta = reference(sc["code"], sc["noise"], p, sc["protocol"])["coincidence"]
    disc = gt["pair_stat_discard"] if mirror else gt["single_stat_discard"]
    depths = PROD_DEPTHS if sc["grid"] == "production" else design_depths(disc, reach=sc["reach"])
    x = np.array(depths, float)
    if not mirror:
        x = np.where(x > 0, x + 1, 0.0)
    spam = dict(prep=True, meas=True, meas_scale=sc["meas_scale"]) if sc["spam"] else None
    rng = np.random.default_rng(sc["seed"])
    est = {k: [] for k in ("block", "free_eps1", "free_beta", "direct_c2")}
    boot_se, acc_errors = [], []
    for rep in range(sc["reps"]):
        P = sm.experiment_dists(p, sc["n_seq"], depths, rng, sc["protocol"], sc["nested"], spam)
        cnt = sm.sample_counts(P, sc["shots"], rng)
        mb = F.mode_fit(x, cnt, d, "block")
        mf = F.mode_fit(x, cnt, d, "free")
        dq = F.direct_quadratic(x, cnt, d)
        est["block"].append(mb["eps1"])
        est["free_eps1"].append(mf["eps1"])
        est["free_beta"].append(mf["beta"])
        est["direct_c2"].append(dq["c2"])
        st = F.accepted_stats(cnt, d)
        acc_errors.append(float((st["n_acc"] * (1 - (st["lam"] * (d - 1) + 1) / d)).sum()))
        if rep < sc["boot_reps"]:
            b = F.bootstrap(lambda c: {"b": F.mode_fit(x, c, d, "block")["eps1"]}, cnt, n_boot=100, seed=rep)
            boot_se.append(b["b"][1])
    row = dict(sc)
    row.update({"depths": " ".join(map(str, depths)), "n_depths": len(depths), "m_max": max(depths),
                "total_shots": sc["shots"] * sc["n_seq"] * len(depths),
                "true_unit_infidelity": true_unit, "coincidence_reference": true_beta,
                "mean_accepted_error_events": float(np.mean(acc_errors))})
    for k, v in est.items():
        v = np.asarray(v, float)
        ref = true_unit if k == "free_eps1" else true_beta
        ok = np.isfinite(v)
        row[f"{k}_mean_over_ref"] = float(np.mean(v[ok]) / ref) if ok.any() else np.nan
        row[f"{k}_sd_over_ref"] = float(np.std(v[ok]) / ref) if ok.any() else np.nan
        row[f"{k}_median_over_ref"] = float(np.median(v[ok]) / ref) if ok.any() else np.nan
    row["block_boot_se_over_ref"] = float(np.mean(boot_se) / true_beta) if boot_se else np.nan
    return row


def scenarios(reps, boot_reps):
    base = dict(code="qubit", noise="si1000", p=5e-3, protocol="mirror", spam=True, meas_scale=1.0,
                grid="designed", reach=1.0, nested=True, n_seq=30, shots=1000, reps=reps, boot_reps=boot_reps)
    out = []

    def add(**kw):
        sc = dict(base)
        sc.update(kw)
        sc["seed"] = 1000 + len(out)
        out.append(sc)

    for code in ("qubit", "qutrit"):
        for p in (2e-3, 5e-3, 1e-2):
            for n_seq in (10, 30, 100):
                for shots in (100, 300, 1000, 3000, 10000):
                    add(code=code, p=p, n_seq=n_seq, shots=shots,
                        boot_reps=boot_reps if (n_seq == 30 and shots == 1000) else 0)
    for code in ("qubit", "qutrit"):
        for shots in (300, 1000, 3000):
            add(code=code, nested=False, shots=shots)                 # independent sequences per depth
            add(code=code, protocol="compiled", shots=shots)          # standard RB closing
            add(code=code, grid="production", shots=shots)            # production depth grid
            add(code=code, reach=0.5, shots=shots)                    # shallower
            add(code=code, reach=2.0, shots=shots)                    # deeper
            add(code=code, meas_scale=4.0, shots=shots)               # readout flip 20 p per qudit
            add(code=code, noise="sd6", p=2e-3, shots=shots)          # uniform noise model
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--boot-reps", type=int, default=30)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    scs = scenarios(a.reps, a.boot_reps)
    for code in ("qubit", "qutrit"):
        for noise in ("si1000", "sd6"):
            model(code, noise)  # make sure the cached tables exist before forking
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows = list(ex.map(run_scenario, scs, chunksize=1))
    path = os.path.join(T.HERE, "results", "finite_shots.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", path, len(rows))


if __name__ == "__main__":
    main()
