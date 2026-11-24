"""The q-general five-moment round against production, and noiseless checks.

A. For 4 sequences of the production five-moment runs (qutrit SI1000, qubit
   SI1000), decode the words with the production generator and rebuild the
   circuits with this study's q-general five-moment profile: .chp text must be
   identical at three depths.
B. At q = 5, 7: 200 random catalogue Cliffords (depth 1) and three deep
   circuits are deterministic (all detectors and the logical 0) without noise.
"""
import os, sys, random
import numpy as np
import qslib, qsim
sys.path.insert(0, qslib.REPO_SCRIPTS)
import verify_default_circuits_unchanged as V
from lrb.dem_simulation import reference_shot

DATA = qslib.DATA
RUNS = {3: "Run-2026-11-19-12-51-34-folded_qutrit_prod5b-single-ancilla-si1000-sdim140",
        2: "Run-2026-11-19-23-39-09-folded_qubit_prod5b-single-ancilla-si1000-sdim140"}
ok = True
for q, run in RUNS.items():
    path = os.path.join(DATA, run)
    depths = [int(x) for x in open(os.path.join(path, "depths.txt")).read().strip().split(",")]
    probs = [float(x) for x in open(os.path.join(path, "probs.txt")).read().strip().split(",")]
    pg = V.generator_for_run(path)
    for seq in range(4):
        words = V.decode_lrb_words(path, pg, seq)
        gen, circs = qsim.lrb_circuits(q, "si1000", "production", words, depths, "five")
        r = V.Renderer(gen)
        same = all(r.text(circs[di], probs[0]).splitlines()[V.HEADER_LINES:]
                   == V._chp(path, "LRB", seq, 0, di).splitlines()[V.HEADER_LINES:]
                   for di in (0, len(depths) // 2, len(depths) - 1))
        print(f"q={q} five-moment seq {seq}: identical to production: {same}", flush=True)
        ok &= same
for q in (5, 7):
    cat = qslib.catalogue(q, "production")
    rng = random.Random(3)
    bad = 0
    for w in rng.sample(cat, 200):
        gen, circs = qsim.lrb_circuits(q, "si1000", "production", [w], [1], "five")
        gen.update_noise_param(circs[0], 0.0)
        _, ref = reference_shot(circs[0])
        bad += int((np.asarray(ref["detectors"]) % q).any() or (np.asarray(ref["logicals"]) % q).any())
    words = [rng.choice(cat) for _ in range(12)]
    gen, circs = qsim.lrb_circuits(q, "sd6", "production", words, [0, 3, 12], "five")
    for c in circs:
        gen.update_noise_param(c, 0.0)
        for _ in range(20):
            _, ref = reference_shot(c)
            bad += int((np.asarray(ref["detectors"]) % q).any() or (np.asarray(ref["logicals"]) % q).any())
    print(f"q={q} five-moment noiseless checks: bad = {bad}")
    ok &= bad == 0
print("ALL OK" if ok else "FAILED")
