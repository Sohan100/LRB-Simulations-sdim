import sys
from lrb.correction.compact_dem import read_run_circuit as read_circuit
from lrb.correction.compact_dem import build_compact_detector_error_model
circ = read_circuit(sys.argv[1])
dem = build_compact_detector_error_model(circ)
nops = len(circ.operations)
for loc in dem.locations:
    for ci, comp in enumerate(loc.component_terms):
        det = [i for (i, s) in comp.detector if s % 3]
        lg = [s for (_, s) in comp.logical if s % 3]
        if not det and lg:
            print(f"op {loc.operation_index}/{nops} gate {loc.gate} qudit {loc.qudit} target {loc.target} component {ci} p={loc.probability:.2e} logical shift {lg}")
