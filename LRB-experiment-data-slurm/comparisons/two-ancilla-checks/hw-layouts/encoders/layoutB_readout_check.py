"""Layout B: each ancilla readout measures exactly one stabilizer (d = 3).

Same test as ../hw_layouts_check.noiseless_tests, on the round with an ancilla
reset after each half (so each readout is the bare syndrome; the simulated round
omits the mid-round reset, which only adds the first readout to the second and
leaves the pass rule unchanged).
"""
import ast
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import hw_layouts_check as H  # noqa: E402

ROUND = ast.literal_eval((HERE / "layoutB_round_ops.txt").read_text())
out, nm = [], 0
for op in ROUND:
    out.append(op)
    if op[0] == "m":
        nm += 1
        if nm == 2:
            out += [("r", 5), ("r", 6)]
for line in H.noiseless_tests("layout B (mid-round reset)", out):
    print(line)
