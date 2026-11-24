"""Noisy memory benchmark of the hardware-layout rounds (production noise).

Same harness as ../noise_bench.py: |+L>, R noisy check rounds (scheduled and
given SI1000/SD6 noise by LRBCircuitGenerator exactly as in the LRB runs),
noiseless X readout; accept if every ancilla readout and the terminal G3, G4
are 0. Usage: hw_layouts_bench.py MODEL P [ROUNDS] [SHOTS]
"""
import pickle
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import noise_bench as NB  # noqa: E402
import protocol_search as PS  # noqa: E402
import hw_layouts_check as H  # noqa: E402
from lrb.code_definitions import FoldedQutritDetectionCode as C9  # noqa: E402


def strip(ops):
    return [op[:2] if op[0] == "m" else op for op in ops]


def schemes(model):
    one, _ = NB.production_round("folded_qutrit", model)
    best = pickle.load(open(HERE.parent / "protocol_search_best.pkl", "rb"))
    out = {"one ancilla (production)": one,
           "conv sites, data relay once": NB.ops_round(best["once"][0][3], 9)}
    for name in ("LEFT  G1,G2(a reads),G3,G3G4^-1(a reads)",
                 "LEFT  G1,G2(a reads),G3,G3G4^-1 hybrid q2-q3",
                 "RIGHT G1(a reads),G2,G3,G3G4^-1 3-anc c via q1",
                 "RIGHT G1(a reads),G2,G3,G4 via q2-q3"):
        out[name] = PS.to_circuit(strip(H.SCHEMES[name]), 9)
    return out


if __name__ == "__main__":
    model, p = sys.argv[1], float(sys.argv[2])
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    shots = int(sys.argv[4]) if len(sys.argv) > 4 else 100000
    for name, ideal in schemes(model).items():
        moments = PS.GEN.schedule_moments(ideal)
        ng = sum(1 for collapse, _ in moments if not collapse)
        cells = []
        for sandwich in (False, True):
            acc, err = NB.memory(C9, ideal, model, p, rounds, sandwich, shots)
            cells.append(f"acc {acc / shots:.4f} err {err / max(acc, 1):.2e} ({err})")
        print(f"{model} p={p} R={rounds} {name:38s} moments {ng}g+{len(moments) - ng}c  "
              f"X: {cells[0]}   Z: {cells[1]}", flush=True)
