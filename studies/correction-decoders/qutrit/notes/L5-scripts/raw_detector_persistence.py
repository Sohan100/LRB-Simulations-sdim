"""How time-local are the repository's raw-outcome detectors?

For one baseline LRB circuit, compile the compact q-ary DEM and, for every unit
Weyl component of every noise location, record which check rounds' raw
detectors it shifts.  Raw detectors = raw ancilla outcome of each round.
"""
import re, sys, time
from collections import Counter
import numpy as np
from sdim.circuit_io import read_circuit
from lrb.dem_simulation import build_compact_detector_error_model

path = sys.argv[1]
t0 = time.time()
circ = read_circuit(path)
dem = build_compact_detector_error_model(circ)
labels = dem.detector_labels
rounds = []
for lab in labels:
    m = re.search(r"_r(\d+)_w(\d+)", lab)
    rounds.append(int(m.group(1)) if m else -1)
rounds = np.array(rounds)
m_final = rounds.max()
print("file", path)
print("detectors", len(labels), "rounds 0..", m_final, "logicals", dem.logical_labels)
print("noise locations (active)", len(dem.locations), "build time %.1fs" % (time.time() - t0))

n_rounds_hist = Counter()
span_hist = Counter()
reaches_final = 0
n_detecting = 0
n_logical_only = 0
n_total = 0
persistent_multi = 0
per_gate = Counter()
per_gate_multi = Counter()
for loc in dem.locations:
    for comp in loc.component_terms:
        n_total += 1
        det_idx = [i for (i, s) in comp.detector if s % 3 != 0]
        flips_logical = any(s % 3 != 0 for (_, s) in comp.logical)
        if not det_idx:
            if flips_logical:
                n_logical_only += 1
            continue
        n_detecting += 1
        rs = sorted(set(rounds[det_idx].tolist()))
        n_rounds_hist[len(rs)] += 1
        span_hist[rs[-1] - rs[0] + 1] += 1
        if rs[-1] == m_final:
            reaches_final += 1
        if rs == list(range(rs[0], m_final + 1)) and len(rs) >= 2:
            persistent_multi += 1
        per_gate[loc.gate] += 1
        if len(rs) > 2:
            per_gate_multi[loc.gate] += 1

print("unit components total", n_total, "detected", n_detecting,
      "undetected-but-logical", n_logical_only)
print("histogram: number of distinct rounds shifted by one unit fault ->", dict(sorted(n_rounds_hist.items())))
print("histogram: time span (rounds) of the shifted raw detectors ->", dict(sorted(span_hist.items())))
print("fraction of detected unit faults whose raw signature reaches the final round m: %.3f" % (reaches_final / max(1, n_detecting)))
print("detected unit faults whose raw signature occupies every round from first appearance to m (and >=2 rounds): %d (%.3f)" % (persistent_multi, persistent_multi / max(1, n_detecting)))
print("faults shifting >2 rounds, by gate:", dict(per_gate_multi), "of", dict(per_gate))
