"""Robustness check under literature-exact SI1000 with ASAP scheduling.

The main checkout switched its qutrit production runs to the moment-scheduled
Gidney et al. SI1000 model, with ancilla resets before each check round. This
script reruns the core qubit analyses under the qubit translation of that
model (``lrb_qubit.circuits_lit``):

``census``  exact first/second-order DEM coefficients (R1, E1, A2, data-2
            floor), in total and per noise component, over sequences and
            depths;
``grid``    sampled LRB-D policies, exact MLD (with |LLR| histograms),
            const0 and const0_mld on identical shots (``experiment.run_point``
            with the literature builders swapped in), plus physical RB at
            p_RB = p and 5p.

    python studies/correction-decoders/qubit/scripts/lit_variant.py census --out <dir>
    python studies/correction-decoders/qubit/scripts/lit_variant.py grid --out <dir> --workers 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from types import SimpleNamespace

_THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
if any(os.environ.get(_var) != "1" for _var in _THREAD_VARS):
    for _var in _THREAD_VARS:
        os.environ[_var] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts" / "qubit512"))

from lrb_qubit import baseline as B  # noqa: E402
from lrb_qubit import circuits_lit as L  # noqa: E402
from lrb_qubit import experiment as E  # noqa: E402

DEPTHS = [0, 2, 4, 6, 10, 14, 18, 20, 22]
P_SMALL = 1e-6
COMPONENTS = ("collapse_idle", "meas_flip", "two_qubit", "reset_flip",
              "idle", "one_qubit", "post_measurement")


def only(component: str) -> L.LiteratureSI1000:
    zero = {name: 0.0 for name in L.LiteratureSI1000.__dataclass_fields__}
    zero[component] = getattr(L.LiteratureSI1000(), component)
    return L.LiteratureSI1000(**zero)


def census(out: Path, num_seq: int) -> None:
    import first_order as FO  # the lead's census helpers

    sequences = E.clifford_sequences(num_seq, max(DEPTHS), 20261028)
    rows = []
    for seq_index, sequence in enumerate(sequences):
        for depth in DEPTHS:
            lc = L.build_lrb_circuit_lit(sequence, depth, P_SMALL,
                                         terminal="data5")
            final = {i for i, (r, _) in enumerate(lc.detector_keys)
                     if r == depth + 1}
            row = {"seq": seq_index, "depth": depth, **FO.coefficients(
                FO.symptom_weights(lc.circuit.detector_error_model()),
                P_SMALL, final)}
            for name in COMPONENTS:
                lc_c = L.build_lrb_circuit_lit(sequence, depth, P_SMALL,
                                               terminal="data5",
                                               noise=only(name))
                part = FO.coefficients(
                    FO.symptom_weights(lc_c.circuit.detector_error_model()),
                    P_SMALL)
                row[f"R1_{name}"] = part["R1"]
                row[f"E1_{name}"] = part["E1"]
            rows.append(row)
    keys = list(rows[0].keys())
    summary = {}
    for depth in DEPTHS:
        sel = [r for r in rows if r["depth"] == depth]
        summary[depth] = {k: [float(np.mean([r[k] for r in sel])),
                              float(np.std([r[k] for r in sel]))]
                          for k in keys if k not in ("seq", "depth")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "lit_first_order_summary.json").write_text(
        json.dumps(summary, indent=1))
    print(f"{'depth':>6s} {'R1':>9s} {'R1_data2':>9s} {'E1':>8s} "
          f"{'U1_data2':>9s} {'A2':>9s}")
    for depth in DEPTHS:
        s = summary[depth]
        print(f"{depth:6d} {s['R1'][0]:9.1f} {s['R1_data2'][0]:9.1f} "
              f"{s['E1'][0]:8.2f} {s['U1_data2'][0]:9.2f} {s['A2'][0]:9.1f}")
    print("\ndepth 22, per component: R1/p, E1/p")
    for name in COMPONENTS:
        s = summary[22]
        print(f"  {name:17s} {s['R1_' + name][0]:9.1f} "
              f"{s['E1_' + name][0]:8.2f}")


def _grid_task(args):
    p, seq_index, sequence, shots, decode_shots = args
    E.Q = SimpleNamespace(
        build_lrb_circuit=lambda *a, **k: L.build_lrb_circuit_lit(*a, **k),
        build_lrb_const0_circuit=(
            lambda *a, **k: L.build_lrb_const0_circuit_lit(*a, **k)))
    result = E.run_point(p, [sequence], DEPTHS, shots,
                         decode_shots=decode_shots,
                         seed=4242 + 1000 * seq_index,
                         noise=L.LiteratureSI1000())
    rb = {}
    for scale in (1.0, 5.0):
        table = B.CountTable.zeros(1, len(DEPTHS))
        for d_index, depth in enumerate(DEPTHS):
            circuit = L.build_physical_rb_circuit_lit(sequence, depth,
                                                      scale * p)
            _, obs = circuit.compile_detector_sampler(
                seed=77 + d_index).sample(shots // 4,
                                          separate_observables=True)
            table.add(0, d_index, shots // 4, shots // 4, int(obs.sum()))
        rb[scale] = table
    return p, seq_index, result, rb


def grid(out: Path, num_seq: int, workers: int, shots: int,
         decode_shots: int, probs: list[float]) -> None:
    sequences = E.clifford_sequences(num_seq, max(DEPTHS), 20261028)
    tasks = [(p, i, sequences[i], shots, decode_shots) for p in probs
             for i in range(num_seq)]
    merged = {p: E.PointResult(p, DEPTHS, num_seq) for p in probs}
    rb_tables = {p: {s: B.CountTable.zeros(num_seq, len(DEPTHS))
                     for s in (1.0, 5.0)} for p in probs}
    with ProcessPoolExecutor(workers) as pool:
        for p, i, result, rb in pool.map(_grid_task, tasks):
            for name, table in result.tables.items():
                target = merged[p].table(name)
                target.shots[i] = table.shots[0]
                target.accepted[i] = table.accepted[0]
                target.errors[i] = table.errors[0]
            for name, hist in result.gap_hist.items():
                merged[p].hist(name)[i] = hist[0]
            for scale, table in rb.items():
                rb_tables[p][scale].shots[i] = table.shots[0]
                rb_tables[p][scale].accepted[i] = table.accepted[0]
                rb_tables[p][scale].errors[i] = table.errors[0]
    out.mkdir(parents=True, exist_ok=True)
    for index, p in enumerate(probs):
        merged[p].to_npz(str(out / f"p{index}.npz"))
        for scale, table in rb_tables[p].items():
            np.savez_compressed(out / f"rb{index}_x{scale:g}.npz",
                                p_rb=scale * p, shots=table.shots,
                                accepted=table.accepted, errors=table.errors)
    (out / "meta.json").write_text(json.dumps(
        {"probs": probs, "depths": DEPTHS, "num_seq": num_seq,
         "shots": shots, "decode_shots": decode_shots,
         "model": "literature SI1000, ASAP moments, reset before checks"},
        indent=1))
    print("grid written to", out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("mode", choices=("census", "grid"))
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-seq", type=int, default=30)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--shots", type=int, default=200_000)
    parser.add_argument("--decode-shots", type=int, default=100_000)
    parser.add_argument("--probs", default="0.0005,0.001,0.002,0.005")
    args = parser.parse_args()
    out = Path(args.out)
    if args.mode == "census":
        census(out, args.num_seq)
    else:
        grid(out, args.num_seq, args.workers, args.shots, args.decode_shots,
             [float(x) for x in args.probs.split(",")])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
