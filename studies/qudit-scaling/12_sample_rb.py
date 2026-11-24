"""Physical single-qudit RB for one configuration, production construction.

30 sequences of catalogue Cliffords (independent of the LRB sequences), the
production mirror construction (one noisy segment per Clifford, ideal
preparation and Z readout), sampled through sdim's DEM with 10,000 shots per
circuit as production does.  Also stores the exact per-sequence polarization
prod_i lambda^(2 n_i) for a check.

    python 12_sample_rb.py --q 5 --noise si1000 --p 0.001,0.002 \
        --depths 0,2,4,6,10,14,18,20,22 --seq-seed 5201 --tag final-prod
Writes cache/<tag>/q<q>-<noise>-<compile>/rb_p<p>.npz with
  pol (n_depths, n_seq) sampled polarization, exact (n_depths, n_seq).
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import qfit
import qsim

SHOTS = 10_000


def job(args):
    q, noise, mode, words, depth, p, seed = args
    gen, circs = qsim.rb_circuits(q, noise, mode, words[:depth], [depth])
    cnt, _ = qsim.sample_rb_circuit(gen, circs[0], p, SHOTS, seed, q)
    return float(qsim.polarization(cnt))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--compile", default="production")
    ap.add_argument("--p", required=True)
    ap.add_argument("--depths", required=True)
    ap.add_argument("--n-seq", type=int, default=30)
    ap.add_argument("--seq-seed", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")]
    out = Path(qsim.qslib.HERE) / "cache" / a.tag / f"q{a.q}-{a.noise}-{a.compile}"
    out.mkdir(parents=True, exist_ok=True)
    seqs = qsim.draw_sequences(a.q, a.compile, a.n_seq, max(depths), a.seq_seed)
    (out / "rb_words.json").write_text(json.dumps(dict(depths=depths, seq_seed=a.seq_seed,
                                                       words=seqs, shots=SHOTS)))
    for p in [float(x) for x in a.p.split(",")]:
        f = out / f"rb_p{p:.6g}.npz"
        if f.exists():
            continue
        jobs, index = [], []
        for s, w in enumerate(seqs):
            for di, d in enumerate(depths):
                jobs.append((a.q, a.noise, a.compile, w, d, p,
                             (a.seq_seed * 7 + s * 1009 + d * 17 + int(p * 1e7)) % 2**63))
                index.append((di, s))
        pol = np.zeros((len(depths), a.n_seq))
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for (di, s), v in zip(index, ex.map(job, jobs, chunksize=8)):
                pol[di, s] = v
        exact = qfit.rb_exact_polarization(a.q, a.noise, seqs, depths, p, a.compile)
        tmp = out / f".rb_p{p:.6g}.tmp.npz"
        np.savez(tmp, pol=pol, exact=exact, p=p, depths=np.array(depths))
        os.replace(tmp, f)
        print(f"RB q={a.q} {a.noise} p={p}: max |sampled-exact| = {np.abs(pol - exact).max():.3f}",
              flush=True)


if __name__ == "__main__":
    main()
