"""Robustness of DEM-based decoders to a miscalibrated noise strength.

Shots are sampled from the true model at ``p``; the decoder's posterior is
computed from a chain model built at ``p_dec = scale * p`` (same structure,
wrong rates). Every rule is scored with the exact posterior of the *true*
model (Rao-Blackwellized), so the numbers isolate the decision quality. The
blip-forgiving rule needs no model and is shown for reference.

Writes ``results/mismatch_check.csv``.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
STUDY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lrb.correction import decoders, forward, sampler, study  # noqa: E402

CACHE = REPO_ROOT / ".cache_correction" / "structures"
SCALES = (0.25, 0.5, 1.0, 2.0, 4.0)
THRESHOLDS = (0.9, 0.97)


def main() -> int:
    rows = []
    for cliff in (0, 1, 2):
        for depth_index, depth in ((4, 10), (8, 22)):
            prepared = study.load_prepared(str(CACHE / f"c{cliff:02d}_d{depth_index}.pkl"))
            for p in (1e-3, 2e-3):
                truth = prepared.structure.at(p)
                rng = np.random.default_rng([cliff, depth, int(p * 1e6)])
                det, _ = sampler.sample_shots(truth.mechanisms, truth.activation, 40000, rng)
                syn = sampler.syndrome_indices(det, truth.layout.detector_index)
                post_true, _ = forward.forward_posteriors_shared(truth, syn)
                bfd = decoders.blip_forgiving_accept(syn)
                rows.append({"cliff": cliff, "depth": depth, "p": p, "scale": "n/a", "rule": "bfd",
                             "acceptance": float(bfd.mean()),
                             "survival": float(post_true[bfd, 0].mean())})
                for scale in SCALES:
                    model = prepared.structure.at(p * scale)
                    post_dec, _ = forward.forward_posteriors_shared(model, syn)
                    l_hat = post_dec.argmax(axis=1)
                    conf = post_dec.max(axis=1)
                    correct = post_true[np.arange(len(l_hat)), l_hat]
                    rows.append({"cliff": cliff, "depth": depth, "p": p, "scale": scale, "rule": "mld",
                                 "acceptance": 1.0, "survival": float(correct.mean())})
                    for t in THRESHOLDS:
                        acc = conf >= t
                        rows.append({"cliff": cliff, "depth": depth, "p": p, "scale": scale, "rule": f"soft{t}",
                                     "acceptance": float(acc.mean()),
                                     "survival": float(correct[acc].mean()) if acc.any() else float("nan")})
                print(f"done c{cliff} m={depth} p={p}", flush=True)
    out = STUDY / "results" / "mismatch_check.csv"
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    # Summary: mean over sequences.
    import collections
    agg = collections.defaultdict(list)
    for row in rows:
        agg[(row["depth"], row["p"], row["rule"], row["scale"])].append((row["acceptance"], row["survival"]))
    for key in sorted(agg, key=lambda k: (k[0], k[1], k[2], str(k[3]))):
        values = np.array(agg[key])
        print(key, "acc=%.4f infid=%.3e" % (values[:, 0].mean(), 1 - np.nanmean(values[:, 1])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
