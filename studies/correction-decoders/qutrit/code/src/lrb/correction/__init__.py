"""Error-correction decoders and analysis tools for qutrit LRB circuits.

The modules in this package turn the compact q-ary detector error model of
one annotated LRB circuit (``lrb.dem_simulation.CompactDetectorErrorModel``)
into a time-ordered chain model, decode syndrome histories with exact
maximum-likelihood inference, and evaluate correction, detection, and hybrid
postselection rules against the LRB-D baseline.

Modules:
    zmod3: Small Z_3 linear-algebra and digit-indexing helpers.
    chain_dem: Mechanism extraction, slot assignment, syndrome transfer maps,
        and per-slot fault kernels.
    forward: Exact forward (filtering) recursion for logical posteriors.
    sampler: Vectorized shot sampler for the extracted mechanism list.
    decoders: Decision rules (LRB-D, MLD, soft-threshold, blip-forgiving,
        hard-decision sliding window).
    metrics: Fidelity metrics and RB decay fits matching the LRB pipeline.
"""
