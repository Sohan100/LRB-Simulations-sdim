"""Validate the general-q circuits and sampler against the production q = 2, 3 runs.

A. Circuit identity: for 6 sequences of each production run, decode the Clifford
   words from the run's own files (production decoder), rebuild the depth-22
   LRB circuit with this study's q-general profile, and compare the rendered
   .chp text with the production file at p-index 0 (byte for byte).
B. Sampler agreement: for the same sequences, sample this study's circuits at two
   p values with the production depths and 1e6 shots, and compare accepted
   counts and polarization per (depth, sequence) with the production
   checkpoint tallies (k = 1, 2, 4): z-scores of acceptance and of the
   logical-error fraction.
Writes results/validation_production.csv.
"""
import os, sys
import numpy as np
import pandas as pd
import qslib, qsim
sys.path.insert(0, qslib.REPO_SCRIPTS)
import verify_default_circuits_unchanged as V

DATA = qslib.DATA
RUNS = {
    3: ("Run-2026-10-28-10-23-30-folded_qutrit-single-ancilla-si1000-sdim140", "si1000"),
    2: ("Run-2026-10-31-10-12-55-folded_qubit-single-ancilla-si1000-sdim140", "si1000"),
}
SEQS = [0, 1, 2, 3, 4, 5]
P_IDX = [6, 9]   # p = 0.005, 0.0125 in the production grid
KS = [1, 2, 4]
rows = []
all_ok = True
for q, (run, noise) in RUNS.items():
    path = os.path.join(DATA, run)
    depths = [int(x) for x in open(os.path.join(path, "depths.txt")).read().strip().split(",")]
    probs = [float(x) for x in open(os.path.join(path, "probs.txt")).read().strip().split(",")]
    prod_gen = V.generator_for_run(path)
    for seq in SEQS:
        words = V.decode_lrb_words(path, prod_gen, seq)
        gen, circs = qsim.lrb_circuits(q, noise, "production", words, depths)
        r = V.Renderer(gen)
        identical = []
        for di in (0, len(depths) // 2, len(depths) - 1):
            mine = r.text(circs[di], probs[0]).splitlines()[V.HEADER_LINES:]
            theirs = V._chp(path, "LRB", seq, 0, di).splitlines()[V.HEADER_LINES:]
            identical.append(mine == theirs)
        ok = all(identical)
        all_ok &= ok
        print(f"q={q} seq {seq}: circuits identical at 3 depths: {ok}", flush=True)
        for pi in P_IDX:
            p = probs[pi]
            prod = {k: np.load(os.path.join(path, "progress", str(pi), ".checkpoint_current",
                                            "unif_check_data", f"{k}.npy")) for k in KS}
            shots = int(open(os.path.join(path, "shots.txt")).read().strip())
            for di, depth in enumerate(depths):
                cnt, rej, _ = qsim.sample_lrb_circuit(gen, circs[di], depth, p, shots,
                                                      seed=1000 * seq + 10 * pi + di, ks=KS, q=q)
                for k in KS:
                    a_mine, a_prod = cnt[k].sum(), prod[k][di, seq].sum()
                    e_mine = 1 - cnt[k][0] / max(a_mine, 1)
                    e_prod = 1 - prod[k][di, seq][0] / max(a_prod, 1)
                    # binomial z-scores (difference of two independent samples)
                    pa = (a_mine + a_prod) / (2 * shots)
                    z_acc = (a_mine - a_prod) / np.sqrt(max(2 * shots * pa * (1 - pa), 1e-12))
                    pe = (e_mine * a_mine + e_prod * a_prod) / max(a_mine + a_prod, 1)
                    se = np.sqrt(max(pe * (1 - pe) * (1 / max(a_mine, 1) + 1 / max(a_prod, 1)), 1e-300))
                    z_err = (e_mine - e_prod) / se if a_mine > 20 and a_prod > 20 and pe > 0 else np.nan
                    rows.append(dict(q=q, seq=seq, p=p, depth=depth, k=k, acc_mine=a_mine,
                                     acc_prod=a_prod, err_mine=e_mine, err_prod=e_prod,
                                     z_acc=z_acc, z_err=z_err))
df = pd.DataFrame(rows)
df.to_csv("results/validation_production.csv", index=False)
for q in RUNS:
    d = df[df.q == q]
    za, ze = d.z_acc.to_numpy(), d.z_err.dropna().to_numpy()
    print(f"q={q}: acceptance z mean {za.mean():+.3f} sd {za.std():.3f} (n={len(za)}); "
          f"logical-error z mean {ze.mean():+.3f} sd {ze.std():.3f} (n={len(ze)})")
print("circuits identical:", all_ok)
