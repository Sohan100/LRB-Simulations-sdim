"""First-order (single-fault) analysis of baseline LRB circuits.

Independent of the chain decoder, this enumerates every single fault (one
location applying one of its Weyl vectors) of a circuit and computes the
leading-order coefficients in p of:

* LRB-D rejection: single faults that flip a used detector;
* undetected logical error: single faults that shift the logical observable
  while leaving every detector trivial (the circuit-level "fault distance 1"
  events), and where in the circuit they sit;
* maximum-likelihood correction failure: single faults are grouped by their
  full detector pattern; within a pattern the decoder outputs the most likely
  logical shift, so the first-order failure is ``sum(pattern) - max(logical
  class)``;
* blip-forgiving detection: which single faults it accepts (isolated
  non-trivial rounds) and whether any of those carries a logical shift.

The coefficients are compared with the exact chain-model LRB-D and sampled
MLD results at small p, where first order dominates.

Usage:
    python scripts/correction/verify_first_order.py --cliffords 0,1,2 --depth-indices 0,2,5,8
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from lrb.correction import chain_dem, decoders, forward, sampler  # noqa: E402

RUN = Path("/pscratch/sd/s/sohan100/LRB-Simulations-sdim/LRB-experiment-data-slurm/"
           "Run-2026-08-04-23-58-49-folded_qutrit-single-ancilla-si1000")
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]


def first_order(mechanisms, layout, p: float) -> dict:
    rounds = layout.num_rounds
    patterns: dict[bytes, np.ndarray] = defaultdict(lambda: np.zeros(3))
    undetected_logical = 0.0
    undetected_by_slot = defaultdict(float)
    rejection_all = 0.0
    rejection_terminal_only = 0.0
    bfd_accepted_logical = 0.0
    bfd_rejection = 0.0
    for mechanism in mechanisms:
        q = mechanism.activation_probability(p)
        weight = q / len(mechanism.vectors)
        per_round = mechanism.detector_effects[:, layout.detector_index]  # (n_v, rounds, 4)
        nontrivial = np.any(per_round != 0, axis=2)  # (n_v, rounds)
        for v in range(len(mechanism.vectors)):
            logical = int(mechanism.logical_effects[v] % 3)
            key = per_round[v].astype(np.int8).tobytes()
            patterns[key][logical] += weight
            if not nontrivial[v].any():
                if logical:
                    undetected_logical += weight
                    undetected_by_slot[f"{mechanism.gate}:{mechanism.channel}:q{mechanism.qudit}"] += weight
                continue
            rejection_all += weight
            if nontrivial[v, -1]:
                rejection_terminal_only += weight
            consecutive = np.any(nontrivial[v, 1:] & nontrivial[v, :-1])
            if nontrivial[v, -1] or consecutive:
                bfd_rejection += weight
            elif logical:
                bfd_accepted_logical += weight
    mld_failure = sum(values.sum() - values.max() for key, values in patterns.items()
                      if any(np.frombuffer(key, dtype=np.int8)))
    return {
        "rejection_all_rounds": rejection_all,
        "rejection_terminal_only": rejection_terminal_only,
        "undetected_logical": undetected_logical,
        "undetected_logical_sources": dict(sorted(undetected_by_slot.items(), key=lambda kv: -kv[1])[:6]),
        "bfd_rejection": bfd_rejection,
        "bfd_accepted_logical": bfd_accepted_logical,
        "mld_failure_first_order": float(mld_failure) + undetected_logical,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cliffords", default="0,1,2")
    parser.add_argument("--depth-indices", default="0,2,5,8")
    parser.add_argument("--p", type=float, default=1e-4)
    parser.add_argument("--shots", type=int, default=200000)
    parser.add_argument("--out", default=str(REPO_ROOT / "docs" / "correction-decoders" / "qutrit-512" /
                                             "results" / "first_order_check.json"))
    args = parser.parse_args()
    from sdim.circuit_io import read_circuit

    from lrb.dem_simulation import build_compact_detector_error_model

    records = []
    for cliff in [int(c) for c in args.cliffords.split(",")]:
        for di in [int(d) for d in args.depth_indices.split(",")]:
            path = RUN / "experiments" / "LRB" / str(cliff) / "4" / f"{di}.chp"
            model = build_compact_detector_error_model(read_circuit(str(path)))
            structure = chain_dem.build_chain_structure(model)
            fo = first_order(structure.mechanisms, structure.layout, args.p)
            chain = structure.at(args.p)
            rounds = chain.num_rounds
            exact_all = decoders.lrbd_exact(chain, np.ones(rounds, bool))
            exact_term = decoders.lrbd_exact(chain, decoders.const_mask(DEPTHS[di], 1))
            exact_bfd = decoders.blip_forgiving_exact(chain)
            rng = np.random.default_rng(cliff * 100 + di)
            det, logical = sampler.sample_shots(chain.mechanisms, chain.activation, args.shots, rng)
            syn = sampler.syndrome_indices(det, chain.layout.detector_index)
            post, _ = forward.forward_posteriors_shared(chain, syn)
            mld_fail = float(1.0 - post.max(axis=1).mean())
            records.append({
                "cliff": cliff, "depth": DEPTHS[di], "p": args.p, **fo,
                "exact_rejection_all_rounds": float(1 - exact_all.sum()),
                "exact_rejection_terminal_only": float(1 - exact_term.sum()),
                "exact_undetected_logical_all_rounds": float(exact_all[1:].sum()),
                "exact_bfd_rejection": float(1 - exact_bfd.sum()),
                "exact_bfd_logical": float(exact_bfd[1:].sum()),
                "sampled_mld_failure": mld_fail,
            })
            r = records[-1]
            print(f"c{cliff} m={DEPTHS[di]:2d}  rej(all) 1st={r['rejection_all_rounds']:.4e} exact={r['exact_rejection_all_rounds']:.4e} | "
                  f"undet-logical 1st={r['undetected_logical']:.3e} exact={r['exact_undetected_logical_all_rounds']:.3e} | "
                  f"MLD fail 1st={r['mld_failure_first_order']:.3e} sampled={r['sampled_mld_failure']:.3e} | "
                  f"BFD logical 1st={r['bfd_accepted_logical']:.1e} exact={r['exact_bfd_logical']:.2e}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(records, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
