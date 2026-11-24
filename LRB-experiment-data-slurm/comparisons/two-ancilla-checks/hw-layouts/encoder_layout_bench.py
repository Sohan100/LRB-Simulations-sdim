"""Noisy 3-round memory benchmark of an encoder-compatible round (same harness as
layout_candidates.bench). Usage:
  encoder_layout_bench.py LAYOUT "i,j,k,l" MODEL P SHOTS [--noreset]
"""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import encoder_layout_rounds as ER  # noqa: E402
import noise_bench as NB  # noqa: E402
import protocol_search as PS  # noqa: E402
from lrb.code_definitions import FoldedQutritDetectionCode as C9  # noqa: E402

if __name__ == "__main__":
    layout, combo, model, p, shots = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
    noreset = "--noreset" in sys.argv
    idx = tuple(int(x) for x in combo.split(","))
    per = ER.candidates(layout, 0)
    ops = ER.grouped(per["G1"][idx[0]], per["G2"][idx[1]], per["G3"][idx[2]], per["G4"][idx[3]])
    ops = [op[:2] if op[0] == "m" else op for op in ops if not (noreset and op[0] == "r")]
    ideal = PS.to_circuit(ops, 9)
    moments = PS.GEN.schedule_moments(ideal)
    ng = sum(1 for collapse, _ in moments if not collapse)
    cells = []
    for sandwich in (False, True):
        acc, err = NB.memory(C9, ideal, model, p, 3, sandwich, shots)
        cells.append(f"acc {acc / shots:.4f} err {err / max(acc, 1):.2e} ({err})")
    print(f"{model} p={p} R=3 encoder layout {layout} {idx}{' no reset' if noreset else ''} "
          f"{ng}g+{len(moments) - ng}c  X: {cells[0]}   Z: {cells[1]}", flush=True)
