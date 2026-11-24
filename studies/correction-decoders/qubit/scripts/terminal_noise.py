"""Effect of a noisy terminal data readout on decoded and postselected LRB.

The qutrit pipeline (and the main qubit grid) keeps the terminal readout
ideal. On hardware the final data measurement carries the same SI1000
measurement noise as the ancillas (``DEPOLARIZE1(p)`` + ``X_ERROR(5p)`` per
measured qubit, after an ``H`` with ``0.1p``). This script compares, on
identical shots of the noisy-readout circuits:

``mld5_tn``          exact MLD with all check rounds and the noisy final readout
``mld5_tn_gapT``     the same, accepting only ``|LLR| >= T``
``lrbd_unif_1_tn``   LRB-D on all check rounds (data-2 semantics)
``lrbd_unif_1_final_tn``  plus the noisy final X-stabilizer check
``const0_tn``        no checks, postselect on the noisy final X stabilizers
``const0_mld_tn``    no checks, decode the noisy final readout

    python studies/correction-decoders/qubit/scripts/terminal_noise.py --out <dir> --workers 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import fits as Fz  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
GAPS = [2.0, 4.0, 6.0]
METHODS = (["mld5_tn", "lrbd_unif_1_tn", "lrbd_unif_1_final_tn", "const0_tn",
            "const0_mld_tn"] + [f"mld5_tn_gap{g:g}" for g in GAPS])


def run_sequence(args):
    p, seq_index, sequence, shots, seed = args
    counts = {m: np.zeros((len(DEPTHS), 2), dtype=np.int64) for m in METHODS}
    for d_index, depth in enumerate(DEPTHS):
        lc = Q.build_lrb_circuit(sequence, depth, p, terminal="data5",
                                 terminal_noise=True)
        dec = M.TransferMatrixMLD(lc.circuit.detector_error_model(),
                                  M.lrb_detector_slots(lc))
        dets, obs = lc.circuit.compile_detector_sampler(
            seed=seed + d_index).sample(shots, separate_observables=True)
        flip = obs[:, 0]
        llr = dec.log_likelihood_ratio(dets)
        wrong = (llr < 0) != flip
        counts["mld5_tn"][d_index] += (shots, wrong.sum())
        for g in GAPS:
            keep = np.abs(llr) >= g
            counts[f"mld5_tn_gap{g:g}"][d_index] += (keep.sum(),
                                                     (keep & wrong).sum())
        raw = lc.raw_syndromes(dets)
        acc = B.accept_mask(raw, lc.rounds, B.used_rounds("unif", 1, depth))
        counts["lrbd_unif_1_tn"][d_index] += (acc.sum(), (acc & flip).sum())
        acc_f = acc & ~dets[:, 4 * lc.num_rounds:].any(axis=1)
        counts["lrbd_unif_1_final_tn"][d_index] += (acc_f.sum(),
                                                    (acc_f & flip).sum())

        lc0 = Q.build_lrb_circuit(sequence, depth, p, terminal="data5",
                                  terminal_noise=True,
                                  forward_check_rounds=(),
                                  terminal_check=False)
        dec0 = M.TransferMatrixMLD(lc0.circuit.detector_error_model(),
                                   M.lrb_detector_slots(lc0))
        dets0, obs0 = lc0.circuit.compile_detector_sampler(
            seed=seed + 500 + d_index).sample(shots, separate_observables=True)
        flip0 = obs0[:, 0]
        acc0 = ~dets0.any(axis=1)
        counts["const0_tn"][d_index] += (acc0.sum(), (acc0 & flip0).sum())
        wrong0 = (dec0.log_likelihood_ratio(dets0) < 0) != flip0
        counts["const0_mld_tn"][d_index] += (shots, wrong0.sum())
    return p, seq_index, counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", required=True)
    parser.add_argument("--probs", default="0.0005,0.001,0.002,0.005,0.01")
    parser.add_argument("--num-seq", type=int, default=30)
    parser.add_argument("--shots", type=int, default=100_000)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    probs = [float(x) for x in args.probs.split(",")]
    sequences = E.clifford_sequences(args.num_seq, max(DEPTHS), 20261028)
    tasks = [(p, i, sequences[i], args.shots, 777 + 1000 * i)
             for p in probs for i in range(args.num_seq)]
    acc = {p: {m: np.zeros((args.num_seq, len(DEPTHS)), np.int64)
               for m in METHODS} for p in probs}
    err = {p: {m: np.zeros((args.num_seq, len(DEPTHS)), np.int64)
               for m in METHODS} for p in probs}
    with ProcessPoolExecutor(args.workers) as pool:
        for p, i, counts in pool.map(run_sequence, tasks):
            for m in METHODS:
                acc[p][m][i] = counts[m][:, 0]
                err[p][m][i] = counts[m][:, 1]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in probs:
        np.savez_compressed(out / f"terminal_noise_p{p:g}.npz",
                            **{f"{m}__accepted": acc[p][m] for m in METHODS},
                            **{f"{m}__errors": err[p][m] for m in METHODS},
                            shots=args.shots, depths=DEPTHS)
        for m in METHODS:
            fit = Fz.bootstrap_fit(DEPTHS, acc[p][m], err[p][m], "lrb",
                                   num_boot=100)
            fid = Fz.per_sequence_fidelity(acc[p][m], err[p][m])
            with np.errstate(invalid="ignore"):
                f_mean = np.nanmean(fid, axis=0)
            rej = 1 - acc[p][m].sum(0) / (args.shots * args.num_seq)
            rows.append({"p": p, "method": m, "f": fit["f"],
                         "f_lo": fit["f_lo"], "f_hi": fit["f_hi"],
                         "F": [float(x) for x in f_mean],
                         "rejection": [float(x) for x in rej]})
            print(f"p={p:<7g} {m:22s} f={fit['f']:.5f} "
                  f"[{fit['f_lo']:.5f},{fit['f_hi']:.5f}] "
                  f"F(0)={f_mean[0]:.4f} F(22)={f_mean[-1]:.4f} "
                  f"rej(22)={rej[-1]:.3f}", flush=True)
    (out / "terminal_noise_summary.json").write_text(json.dumps(rows,
                                                               indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
