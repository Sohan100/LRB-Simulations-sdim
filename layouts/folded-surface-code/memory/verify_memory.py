"""Checks on the memory circuits before any sampling.

1. Noiseless circuits: every detector and the observable are deterministic (zero in all shots).
2. Noisy circuits: the detector error model decomposes into graphlike errors and its shortest
   graphlike logical error has weight d (the schedule keeps the full circuit distance).
3. Folded code: every two-qubit gate of the circuit acts on a coupler of the folded layout
   (data/layout-d<d>.json written by gen_layout.py).
4. Prints qubits and moments per round for each code and noise model.
"""
import json
import os
import sys


HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import memory_circuits as mc  # noqa: E402

LAYOUT_DATA = os.path.join(HERE, "..", "data")


def coupler_check(d: int) -> int:
    lay = json.load(open(os.path.join(LAYOUT_DATA, "layout-d%d.json" % d)))
    qid = {tuple(q["unfolded"]): "q%d" % q["q"] for q in lay["data_qubits"]}
    aid = {tuple(a["unfolded"]): "a%d" % a["a"] for a in lay["ancillas"]}
    couplers = {frozenset(c["between"]) for c in lay["couplers"]}
    code = mc.folded_surface_code(d)
    n = 0
    for c in code.checks:
        for q in c.schedule:
            if q is None:
                continue
            assert frozenset((aid[c.ancilla], qid[q])) in couplers, (d, c.ancilla, q)
            n += 1
    # and the check of every ancilla is the one in the layout file
    for a in lay["ancillas"]:
        chk = [c for c in code.checks if c.ancilla == tuple(a["unfolded"])][0]
        assert chk.basis == a["check"]
        assert sorted(int(qid[q][1:]) for q in chk.schedule if q is not None) == a["data"]
    return n


def main(dmax: int = 7) -> None:
    for d in range(3, dmax + 1, 2):
        for code in ("folded", "rotated"):
            for noise in ("SD6", "SI1000"):
                for basis in ("X", "Z"):
                    for gate in ((None, "H", "S") if code == "folded" else (None,)):
                        if gate == "S" and basis == "X":
                            continue
                        for rounds in ((d, d + 2) if gate == "S" else (d, d + 1)):
                            clean = mc.memory_circuit(
                                mc.CODES[code](d), rounds, basis,
                                mc.NoiseModel.noiseless("CZ" if noise == "SI1000" else "CX"), gate)
                            # the detector sampler randomizes the gauge, so a detector or observable
                            # that is not deterministic shows up as random flips here
                            det, obs = clean.compile_detector_sampler().sample(256, separate_observables=True)
                            assert not det.any() and not obs.any(), (code, d, noise, basis, gate, rounds)
                            clean.detector_error_model()  # raises on a non-deterministic detector
                        for det_mode in ("all", "sector"):
                            if gate == "S" and det_mode == "all":
                                continue
                            circ = mc.build(code, d, noise, 0.001, basis, logical_gate=gate, detectors=det_mode)
                            dem = circ.detector_error_model(decompose_errors=True)
                            w = len(dem.shortest_graphlike_error())
                            assert w == d, (code, d, noise, basis, gate, det_mode, w)
            n = mc.CODES[code](d).num_qubits
            print("d=%d %-8s qubits %4d  moments/round SD6 %d SI1000 %d  distance d (X, Z; plain/H/S memories)"
                  % (d, code, n, mc.round_depth(code, "SD6"), mc.round_depth(code, "SI1000")))
        print("      folded: all %d two-qubit gates per round lie on layout couplers" % coupler_check(d))
    print("all memory-circuit checks passed")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 7)
