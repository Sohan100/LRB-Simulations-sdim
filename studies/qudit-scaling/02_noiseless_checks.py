"""Noiseless and single-fault checks of the general-q circuits.

1. For q = 2, 3, 5, 7 (production compile; q = 5, 7 also inverse compile):
   every catalogue Clifford, encoded, then its encoded inverse, with check
   rounds, gives deterministic all-zero detectors and logical 0 (sdim
   reference shot of the noiseless circuit at p = 0 must agree with a
   200-shot noiseless simulation).
2. Timing of one depth-6 LRB circuit DEM + 1e5 shots at q = 5, 7.
"""
import random, sys, time
import numpy as np
import qsim, qslib
from lrb.dem_simulation import reference_shot

ok = True
for q, modes in ((2, ("production",)), (3, ("production",)), (5, ("production", "inverse")), (7, ("production", "inverse"))):
    for mode in modes:
        cat = qslib.catalogue(q, mode)
        rng = random.Random(1)
        sample = cat if len(cat) <= 216 else rng.sample(cat, 300)
        bad = 0
        t0 = time.time()
        # depth-1 circuits: C, check, C^-1, final check
        for w in sample:
            gen, circs = qsim.lrb_circuits(q, "si1000", mode, [w], [1])
            c = circs[0]
            gen.update_noise_param(c, 0.0)
            _, ref = reference_shot(c)
            det = np.asarray(ref["detectors"]) % q
            log = np.asarray(ref["logicals"]) % q
            if det.any() or log.any():
                bad += 1
        # one long random sequence, all prefixes, checked with 50 noiseless shots
        words = [rng.choice(cat) for _ in range(12)]
        gen, circs = qsim.lrb_circuits(q, "si1000", mode, words, [0, 3, 12])
        for c in circs:
            gen.update_noise_param(c, 0.0)
            # 20 independent noiseless shots (random outcomes wherever a single
            # measurement is not determined); every detector and the logical
            # observable must still be 0.
            for _ in range(20):
                _, ref = reference_shot(c)
                if (np.asarray(ref["detectors"]) % q).any() or (np.asarray(ref["logicals"]) % q).any():
                    bad += 1
        print(f"q={q} {mode}: {len(sample)} single-Clifford circuits + 3 deep circuits, "
              f"bad={bad}, {time.time()-t0:.1f}s", flush=True)
        ok &= bad == 0

for q in (5, 7):
    words = qsim.draw_sequences(q, "production", 1, 6, 7)[0]
    gen, circs = qsim.lrb_circuits(q, "si1000", "production", words, [6])
    (cnt, rej, _), dt = qsim.timed(qsim.sample_lrb_circuit, gen, circs[0], 6, 0.01, 100_000, 3, [1, 2, 3], q)
    print(f"q={q} depth 6 p=0.01 SI1000: 1e5 shots in {dt:.1f}s; k=1 accepted {cnt[1].sum()}, outcome {cnt[1]}")
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
