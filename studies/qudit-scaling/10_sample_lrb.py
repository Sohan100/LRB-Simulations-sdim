"""Sample LRB-D (all k = 1..6 and const1 from the same shots) for one configuration.

    python 10_sample_lrb.py --q 5 --noise si1000 --compile production \
        --p 0.01,0.02 --depths 0,1,2,3,4,6 --shots 200000 --n-seq 30 \
        --seq-seed 501 --tag scan --workers 12

Writes one file per p: cache/<tag>/q<q>-<noise>-<compile>/p<p>.npz with
  unif   (6, n_depths, n_seq, q)  accepted tallies for k = 1..6
  urej   (6, n_depths, n_seq)     rejected counts
  c1     (n_depths, n_seq, q), c1rej   final-round-only postselection
and words.json with the Clifford words (the sequences are nested: depth m
uses the first m words, as in production).  Existing p files are skipped, so
the script resumes.  Seeds are deterministic in (seq-seed, p, seq, depth).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import qsim

KS = (1, 2, 3, 4, 5, 6)


def job(args):
    q, noise, mode, words, depth, p, shots, seed, round_name = args
    gen, circs = qsim.lrb_circuits(q, noise, mode, words[:depth], [depth], round_name)
    cnt, rej, _ = qsim.sample_lrb_circuit(gen, circs[0], depth, p, shots, seed, list(KS), q)
    return cnt, rej


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", type=int, required=True)
    ap.add_argument("--noise", required=True)
    ap.add_argument("--compile", default="production")
    ap.add_argument("--round", default="seven", choices=("seven", "five"),
                    help="one-ancilla check round: published seven-moment or five-moment")
    ap.add_argument("--p", required=True)
    ap.add_argument("--depths", default="0,1,2,3,4,6")
    ap.add_argument("--shots", type=int, default=200_000)
    ap.add_argument("--shots-per-depth", default=None,
                    help="comma list aligned with --depths; overrides --shots per depth")
    ap.add_argument("--chunk", type=int, default=0,
                    help="split every circuit into chunks of at most this many shots "
                         "(0 = one job per circuit); chunk seeds are derived from the circuit seed")
    ap.add_argument("--n-seq", type=int, default=30)
    ap.add_argument("--seq-seed", type=int, required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    depths = [int(x) for x in a.depths.split(",")]
    probs = [float(x) for x in a.p.split(",")]
    suffix = "" if a.round == "seven" else "-five"
    out = Path(qsim.qslib.HERE) / "cache" / a.tag / f"q{a.q}-{a.noise}-{a.compile}{suffix}"
    out.mkdir(parents=True, exist_ok=True)
    seqs = qsim.draw_sequences(a.q, a.compile, a.n_seq, max(depths), a.seq_seed)
    shots_d = ([int(float(x)) for x in a.shots_per_depth.split(",")] if a.shots_per_depth
               else [a.shots] * len(depths))
    assert len(shots_d) == len(depths)
    meta = dict(q=a.q, noise=a.noise, compile=a.compile, round=a.round, depths=depths, shots=a.shots,
                shots_per_depth=shots_d,
                n_seq=a.n_seq, seq_seed=a.seq_seed, ks=list(KS), words=seqs)
    wpath = out / "words.json"
    if wpath.exists():
        old = json.loads(wpath.read_text())
        assert old["words"] == seqs and old["depths"] == depths, \
            "existing folder has different settings"
        if old.get("shots_per_depth", [old["shots"]] * len(depths)) != shots_d:
            # a different shot schedule is allowed per p file; it is recorded in the file
            print("note: shot schedule differs from words.json; recorded per p file", flush=True)
    else:
        wpath.write_text(json.dumps(meta))
    for p in probs:
        f = out / f"p{p:.6g}.npz"
        if f.exists():
            print(f"skip {f}", flush=True)
            continue
        t0 = time.time()
        jobs, index = [], []
        pi = int(round(p * 1e7))
        for s, words in enumerate(seqs):
            for di, d in enumerate(depths):
                seed = (a.seq_seed * 1_000_003 + pi * 7919 + s * 104_729 + d * 13) % (2**63)
                n = shots_d[di]
                if a.chunk and n > a.chunk:
                    n_chunks = -(-n // a.chunk)
                    for c in range(n_chunks):
                        part = min(a.chunk, n - c * a.chunk)
                        jobs.append((a.q, a.noise, a.compile, words, d, p, part,
                                     (seed * 31 + 1 + c) % (2**63), a.round))
                        index.append((di, s))
                else:
                    jobs.append((a.q, a.noise, a.compile, words, d, p, n, seed, a.round))
                    index.append((di, s))
        unif = np.zeros((len(KS), len(depths), a.n_seq, a.q), np.int64)
        urej = np.zeros((len(KS), len(depths), a.n_seq), np.int64)
        c1 = np.zeros((len(depths), a.n_seq, a.q), np.int64)
        c1rej = np.zeros((len(depths), a.n_seq), np.int64)
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for (di, s), (cnt, rej) in zip(index, ex.map(job, jobs, chunksize=1)):
                for ki, k in enumerate(KS):
                    unif[ki, di, s] += cnt[k]
                    urej[ki, di, s] += rej[k]
                c1[di, s] += cnt["c1"]
                c1rej[di, s] += rej["c1"]
        tmp = out / f".p{p:.6g}.tmp.npz"
        np.savez(tmp, unif=unif, urej=urej, c1=c1, c1rej=c1rej, p=p,
                 depths=np.array(depths), shots=a.shots, shots_per_depth=np.array(shots_d))
        os.replace(tmp, f)
        print(f"q={a.q} {a.noise} {a.compile} p={p}: {time.time() - t0:.0f}s, "
              f"k=1 acceptance by depth {unif[0].sum(axis=(1, 2)) / (np.array(shots_d) * a.n_seq)}",
              flush=True)


if __name__ == "__main__":
    main()
