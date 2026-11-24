"""End-to-end example of the recommended analysis on a synthetic hardware-like run.

Qubit code, SI1000 at p = 0.005, noisy encoder and readout, mirror closing,
100 random sequences (independent at every depth) x 300 shots x 9 depths
(2.7e5 shots). Prints the estimates next to the exact answers.

On hardware, replace the synthetic counts by
    counts[circuit, depth, tcfit.record_to_outcome(x_bits, d)]
built from the five X-basis data readouts of every shot.
"""
from __future__ import annotations

import numpy as np

import tcfit as F
import tcsteps as S
from tcref import design_depths, reference

CODE, NOISE, P = "qubit", "si1000", 5e-3


def main():
    sm = S.StepModel(CODE, NOISE)
    ref = reference(CODE, NOISE, P)
    depths = design_depths(ref["discard"])
    rng = np.random.default_rng(2026)
    probs = sm.experiment_dists(P, 100, depths, rng, nested=False, spam=dict(prep=True, meas=True, meas_scale=1.0))
    counts = sm.sample_counts(probs, 300, rng)
    res = F.analyze(depths, counts, sm.d, f_cov=ref["f_cov"])
    print(f"depths {depths}, {counts.sum()} shots")
    print(f"block-rate fit        {res['eps_block'][0]:.3e} +- {res['eps_block'][1]:.1e}   (exact coincidence part {ref['coincidence']:.3e})")
    print(f"  x F_cov = {ref['f_cov']:.3f}     {res['eps_block_cov'][0]:.3e} +- {res['eps_block_cov'][1]:.1e}   (exact logical pair infidelity {ref['true_unit']:.3e})")
    print(f"free-rate fit         {res['eps_free'][0]:.3e} +- {res['eps_free'][1]:.1e}   (same target)")
    print(f"discard per pair      {res['discard'][0]:.4f} +- {res['discard'][1]:.4f}   (exact {ref['discard']:.4f})")
    print("depth  acceptance  accepted infidelity observed +- se   predicted by the block fit")
    for i, m in enumerate(depths):
        print(f"{m:5d}  {res['accept_observed'][i]:.3f}       {res['eps_observed'][i]:.2e} +- {res['eps_observed_se'][i]:.1e}"
              f"            {res['eps_predicted'][i]:.2e}")


if __name__ == "__main__":
    main()
