"""Exact two-fault expansion of the k = 1 accepted logical error versus q.

For each (q, noise, compile, depth m) and a set of random Clifford sequences,
build the production LRB-D circuit (mirror inverse, check after every forward
Clifford, all rounds postselected), take sdim's DEM without merging, and
compute with qpairs:

  S1     first-order coefficient (only faults after the last check that flip
         the ideal logical readout: depth-independent, part of the SPAM
         amplitude b)
  S2b    sum over pairs of *detectable* mechanisms of (pi_i pi_j / p^2) a_ij,
         the two-fault silent-logical coefficient, with
         a_ij = (q-1) q^-w_ij exactly; split by the rank w_ij and by where
         the two faults sit:
           F = forward Clifford gates, C = forward check rounds,
           I = inverse (mirror) Clifford gates, E = final check round,
         plus whether both sit in the same "window" between postselected
         checks.

Low-p accepted logical error at depth m is eps(m) = S1 p + S2(m) p^2 + O(p^3).
Writes results/pair_expansion_<tag>.csv.gz (one row per sequence, depth, class).

    python 20_pair_expansion.py --q 2,3,5,7,11 --noise si1000,sd6 \
        --compile production,inverse --depths 1,2,3,4 --n-seq 12 --workers 12
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

import qpairs
import qsim
import qslib

P_REF = 1e-4  # any p in the linear regime of pi(p); results are divided by p^n


def build_with_segments(q, noise, mode, words, depth):
    """The production circuit and the op-index range of every segment."""
    from sdim.circuit import Circuit

    gen = qslib.generator(q, noise, mode)
    cd = gen.code_definition
    fwd, inv_all = gen._clifford_and_inverse_gate_lists(list(words))
    inv = gen._closing_gate_lists(list(words), inv_all, len(words), depth) if depth else []
    segs = []
    c = Circuit(dimension=q, num_qudits=cd.encoded_num_qudits)
    c = c + cd.logical_plus_initial_state()
    segs.append(("P", 0, len(c.operations)))

    def add(label, sub):
        nonlocal c
        start = len(c.operations)
        c = c + sub
        segs.append((label, start, len(c.operations)))

    for i in range(depth):
        add(f"F{i}", gen._noisy_segment(gen._logical_clifford_circuit(fwd[i]), True))
        add(f"C{i}", gen._noisy_segment(gen._check_round_circuit(), True))
    for i, g in enumerate(inv):
        add(f"I{i}", gen._noisy_segment(gen._logical_clifford_circuit(g), True))
    add("E", gen._noisy_segment(gen._check_round_circuit(), True))
    add("T", cd.terminal_measurement())
    gen.append_lrb_detector_events(c, depth)
    # same circuit as the production generator builds
    _, ref_circs = qsim.lrb_circuits(q, noise, mode, list(words), [depth])
    assert len(ref_circs[0].operations) == len(c.operations)
    return gen, c, segs


_SRC = re.compile(r"@(\d+):((?:q\d+,?)+)")


def classifier(segs):
    starts = np.array([s[1] for s in segs])
    labels = [s[0] for s in segs]

    def where(src):
        m = _SRC.search(src)
        idx = int(m.group(1))
        wires = [int(w[1:]) for w in m.group(2).split(",") if w]
        seg = labels[int(np.searchsorted(starts, idx, side="right") - 1)]
        anc = any(w >= 5 for w in wires)
        kind = "N2" if src.startswith("N2") else ("flip" if "[f]" in src or "[p]" in src else "dep1")
        return seg, anc, kind

    def classify(src_i, src_j):
        si, ai, ki = where(src_i)
        sj, aj, kj = where(src_j)
        ti, tj = si[0], sj[0]
        pair = "".join(sorted([ti, tj]))
        # same window between postselected checks (k = 1): F_i with C_i;
        # the whole inverse block I* with E.
        def window(s):
            if s[0] in "FC":
                return "w" + s[1:]
            return "inv"
        same = window(si) == window(sj)
        return dict(seg_pair=pair, same_window=same,
                    anc_pair="".join(sorted(["a" if ai else "d", "a" if aj else "d"])),
                    kind_pair="+".join(sorted([ki, kj])))

    return classify


def job(args):
    q, noise, mode, words, depth, seq = args
    from sdim.dem import DetectorErrorModel
    from lrb.lrb_simulation import NoiseModelUtils

    gen, c, segs = build_with_segments(q, noise, mode, words[:depth], depth)
    gen.update_noise_param(c, P_REF)
    NoiseModelUtils.ensure_noise_params(c)
    dem = DetectorErrorModel.from_circuit(c, merge=False)
    res = qpairs.second_order(dem, p=P_REF)
    base = dict(q=q, noise=noise, compile=mode, depth=depth, seq=seq,
                S1=res["S1"], S2b_total=res["S2b"], n_mech=res["n_mech"],
                tokens=sum(qslib.token_count(w) for w in words[:depth]))
    df = res["pairs"]
    if not len(df):
        return [dict(base, w=-1, seg_pair="", anc_pair="", kind_pair="", n_pairs=0, S2b=0.0)]
    where = classifier_table(segs, res["sources"])
    seg = where["seg"].to_numpy(); anc = where["anc"].to_numpy(); kind = where["kind"].to_numpy()
    si, sj = seg[df.i.to_numpy()], seg[df.j.to_numpy()]
    df = df.assign(
        seg_pair=["".join(sorted((x, y))) for x, y in zip(si, sj)],
        anc_pair=["".join(sorted((x, y))) for x, y in zip(anc[df.i], anc[df.j])],
        kind_pair=["+".join(sorted((x, y))) for x, y in zip(kind[df.i], kind[df.j])],
        contrib=df.weight * df.a)
    rows = []
    for keys, g in df.groupby(["w", "seg_pair", "anc_pair", "kind_pair"]):
        w, sp, ap, kp = keys
        rows.append(dict(base, w=int(w), seg_pair=sp, anc_pair=ap, kind_pair=kp,
                         n_pairs=len(g), S2b=float(g["contrib"].sum())))
    return rows


def classifier_table(segs, sources):
    """Segment letter (F, C, I, E), data/ancilla and noise kind per mechanism."""
    starts = np.array([s[1] for s in segs])
    labels = [s[0][0] for s in segs]
    out = []
    for src in sources:
        m = _SRC.search(src)
        idx = int(m.group(1))
        wires = [int(w[1:]) for w in m.group(2).split(",") if w]
        seg = labels[int(np.searchsorted(starts, idx, side="right") - 1)]
        anc = "a" if any(w >= 5 for w in wires) else "d"
        kind = "N2" if src.startswith("N2") else ("flip" if ("[f]" in src or "[p]" in src) else "dep1")
        out.append((seg, anc, kind))
    return pd.DataFrame(out, columns=["seg", "anc", "kind"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default="2,3,5,7")
    ap.add_argument("--noise", default="si1000,sd6")
    ap.add_argument("--compile", default="production")
    ap.add_argument("--depths", default="1,2,3,4")
    ap.add_argument("--n-seq", type=int, default=12)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tag", default="main")
    a = ap.parse_args()
    jobs = []
    depths = [int(x) for x in a.depths.split(",")]
    for q in [int(x) for x in a.q.split(",")]:
        for mode in a.compile.split(","):
            if q == 2 and mode == "inverse":
                continue
            seqs = qsim.draw_sequences(q, mode, a.n_seq, max(depths), 7000 + q)
            for noise in a.noise.split(","):
                for s, words in enumerate(seqs):
                    for d in depths:
                        jobs.append((q, noise, mode, words, d, s))
    rows = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(job, jobs, chunksize=1)):
            rows.extend(r)
            if i % 50 == 0:
                print(f"{i + 1}/{len(jobs)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_csv(f"results/pair_expansion_{a.tag}.csv.gz", index=False)
    print(out.groupby(["q", "noise", "compile", "depth"])["S2b"].sum().unstack("depth")
          / a.n_seq)


if __name__ == "__main__":
    main()
