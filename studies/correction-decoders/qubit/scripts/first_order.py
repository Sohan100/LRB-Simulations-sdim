"""Exact low-order structure of the qubit [[5,1,2]] LRB detector error model.

To leading order in p only single faults matter. Group the DEM mechanisms by
their detector set ``s`` and let ``P0(s)`` (``P1(s)``) be the summed
probability of mechanisms with that symptom that leave (flip) the logical
observable. Then, with every check round and the final data readout used:

* first-order rejection of LRB-D             ``R1  = sum_s (P0 + P1)``
* first-order error of exact MLD correction   ``E1  = sum_s min(P0, P1)``
  (plus undetectable logical mechanisms, symptom ``s = {}``)
* second-order accepted error of LRB-D        ``A2  = sum_s P0 * P1``
  (two faults with the same symptom cancel it; the logical flips differ)

These are exact coefficients of the respective probability expansions. The
script reports them per Clifford sequence and depth for the data5 circuit, and
classifies the first-order weight by the fault's physical origin
(measurement/reset flips, idle, two-qubit gates, one-qubit gates) by building
DEMs with one SI1000 component switched on at a time.

Example:
    python studies/correction-decoders/qubit/scripts/first_order.py --num-seq 30 --out <dir>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from lrb_qubit import circuits as Q  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402
from lrb_qubit import mld as M  # noqa: E402

P_SMALL = 1e-6
DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
COMPONENTS = {
    "meas_flip": dict(meas_flip=1.0),
    "reset_flip": dict(reset_flip=1.0),
    "meas_idle": dict(meas_idle=1.0),
    "reset_idle": dict(reset_idle=1.0),
    "meas_depol": dict(meas_depol=1.0),
    "two_qubit": dict(two_qubit=1.0),
    "one_qubit": dict(one_qubit=1.0),
}


def only(component: str) -> Q.SI1000:
    """SI1000 with every scale factor zero except one component."""
    zero = {name: 0.0 for name in Q.SI1000.__dataclass_fields__}
    base = Q.SI1000()
    zero[component] = getattr(base, component)
    return Q.SI1000(**zero)


def symptom_weights(dem) -> dict[tuple, list[float]]:
    """``{detector tuple: [P0, P1]}`` from single mechanisms."""
    weights: dict[tuple, list[float]] = defaultdict(lambda: [0.0, 0.0])
    for mech in M.dem_mechanisms(dem):
        weights[mech.detectors][int(mech.flips_observable)] += mech.probability
    return weights


def coefficients(weights: dict[tuple, list[float]], p: float,
                 final_detectors: set[int] | None = None) -> dict[str, float]:
    """Leading-order coefficients divided by the appropriate power of p.

    ``final_detectors`` (if given) lists detector indices that come from the
    terminal data readout. Symptoms made only of those detectors are seen by
    LRB-D with the data-5 readout but not with the repo's data-2 readout.
    """
    r1 = e1 = a2 = undetected = 0.0
    r1_data2 = undetected_data2 = 0.0
    for symptom, (p0, p1) in weights.items():
        if not symptom:
            undetected += p1
            undetected_data2 += p1
            continue
        r1 += p0 + p1
        e1 += min(p0, p1)
        a2 += p0 * p1
        if final_detectors is not None:
            if set(symptom) <= final_detectors:
                undetected_data2 += p1
            else:
                r1_data2 += p0 + p1
    out = {"R1": r1 / p, "E1": (e1 + undetected) / p, "A2": a2 / p ** 2,
           "U1": undetected / p}
    if final_detectors is not None:
        out["R1_data2"] = r1_data2 / p
        out["U1_data2"] = undetected_data2 / p
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--num-seq", type=int, default=30)
    parser.add_argument("--seq-seed", type=int, default=20261028)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    sequences = E.clifford_sequences(args.num_seq, max(DEPTHS), args.seq_seed)
    rows = []
    for seq_index, sequence in enumerate(sequences):
        for depth in DEPTHS:
            lc = Q.build_lrb_circuit(sequence, depth, P_SMALL,
                                     terminal="data5")
            final = {i for i, (r, _) in enumerate(lc.detector_keys)
                     if r == depth + 1}
            total = coefficients(
                symptom_weights(lc.circuit.detector_error_model()),
                P_SMALL, final)
            row = {"seq": seq_index, "depth": depth, **total}
            for name, _ in COMPONENTS.items():
                lc_c = Q.build_lrb_circuit(sequence, depth, P_SMALL,
                                           terminal="data5", noise=only(name))
                part = coefficients(
                    symptom_weights(lc_c.circuit.detector_error_model()),
                    P_SMALL)
                row[f"R1_{name}"] = part["R1"]
                row[f"E1_{name}"] = part["E1"]
            rows.append(row)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = list(rows[0].keys())
    with open(out / "first_order_per_sequence.csv", "w") as handle:
        handle.write(",".join(keys) + "\n")
        for row in rows:
            handle.write(",".join(str(row[k]) for k in keys) + "\n")

    summary = {}
    for depth in DEPTHS:
        sel = [r for r in rows if r["depth"] == depth]
        summary[depth] = {k: [float(np.mean([r[k] for r in sel])),
                              float(np.std([r[k] for r in sel]))]
                          for k in keys if k not in ("seq", "depth")}
    (out / "first_order_summary.json").write_text(json.dumps(summary,
                                                             indent=1))
    header = ("depth", "R1", "R1_data2", "E1", "U1_data2", "A2")
    print(" ".join(f"{h:>10s}" for h in header))
    for depth in DEPTHS:
        s = summary[depth]
        print(f"{depth:>10d} " + " ".join(
            f"{s[h][0]:10.2f}" for h in header[1:]))
    comp = ("meas_flip", "reset_flip", "meas_idle", "reset_idle",
            "meas_depol", "two_qubit", "one_qubit")
    print("\nfirst-order MLD error coefficient E1 by SI1000 component "
          "(mean over sequences):")
    print(" ".join(f"{c:>11s}" for c in ("depth",) + comp))
    for depth in DEPTHS:
        s = summary[depth]
        print(f"{depth:>11d} " + " ".join(
            f"{s['E1_' + c][0]:11.2f}" for c in comp))
    print("\nfirst-order rejection R1 by component:")
    for depth in DEPTHS:
        s = summary[depth]
        print(f"{depth:>11d} " + " ".join(
            f"{s['R1_' + c][0]:11.2f}" for c in comp))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
