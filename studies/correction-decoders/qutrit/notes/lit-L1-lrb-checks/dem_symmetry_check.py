"""Check, on actual baseline LRB circuits, that the accepted logical-shift distribution
is symmetric under k -> -k (so sum_k w^k P(k) is real), for the 'unif 1' (all rounds)
and 'const 1' (terminal round only) acceptance rules. Read-only use of the run folder.
usage: python -I dem_symmetry_check.py <worktree_root> <circuit_path> <shots>"""
import sys
import time
import numpy as np

root, cpath, shots = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path.insert(0, root + "/src")
from lrb.correction.compact_dem import read_run_circuit as read_circuit                      # noqa: E402
from lrb.correction.compact_dem import build_compact_detector_error_model, sample_detector_error_model  # noqa: E402

t0 = time.time()
c = read_circuit(cpath)
model = build_compact_detector_error_model(c)
t1 = time.time()
payload = sample_detector_error_model(model, extra_shots=shots)
t2 = time.time()
labels = [d["label"] for d in payload["detectors"]]
D = np.stack([d["data"] for d in payload["detectors"]], axis=1) % 3
L = np.stack([d["data"] for d in payload["logicals"]], axis=1)[:, 0] % 3
rounds = sorted({int(lab.split("_r")[1].split("_")[0]) for lab in labels})
term = max(rounds)
term_cols = [i for i, lab in enumerate(labels) if f"_r{term}_" in lab]
print(f"build {t1-t0:.1f}s sample {t2-t1:.1f}s; detectors={len(labels)} rounds={rounds}")
w = np.exp(2j * np.pi / 3)
for name, cols in (("unif 1 (all rounds)", list(range(len(labels)))), ("const 1 (terminal)", term_cols)):
    acc = np.all(D[:, cols] == 0, axis=1)
    n = acc.sum()
    P = np.array([(L[acc] == k).sum() for k in range(3)]) / max(n, 1)
    z = (P * w ** np.arange(3)).sum()
    se = np.sqrt(P[1] * (1 - P[1]) / max(n, 1))
    print(f"{name}: accepted {n}/{shots} ({n/shots:.4f}); P(shift)={np.round(P,5)}; "
          f"P(1)-P(2)={P[1]-P[2]:+.2e} (single-bin s.e. ~{se:.1e}); sum w^k P(k) = {z.real:.5f}{z.imag:+.5f}i")
