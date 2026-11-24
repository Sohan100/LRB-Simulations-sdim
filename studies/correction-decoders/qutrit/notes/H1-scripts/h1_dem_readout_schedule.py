"""Sequential vs parallel ancilla readout, SI1000 vs hardware-calibrated noise.

Builds the repo's folded_qutrit LRB circuits (single-ancilla profile) with the
repo generator, plus a variant whose check round applies all twelve
ancilla couplings first and then measures ancillas 5-8 in ONE measurement
layer (the generator groups consecutive M gates into one SI1000 layer). Each
circuit is compiled once into the repo's compact DEM; the noise locations are
then re-weighted for several noise models without recompiling (the DEM
responses do not depend on the probabilities).

Location classes (from the generator's prob_scale/channel tags):
  2q            N2 after CNOT/CZ/SWAP          SI1000: p
  1q            N1 d after 1-qutrit gates      SI1000: 0.1 p
  idle_meas     N1 d on wires idle in a measurement layer   SI1000: 2p
  idle_reset    N1 d on data during the reset layer         SI1000: 2p
  meas_depol    N1 d on the measured ancilla   SI1000: p
  meas_flip     N1 f before M                  SI1000: 5p
  reset_flip    N1 f after RESET               SI1000: 2p
Hardware-calibrated model "HW-A" (see H1-sc-hardware.md for sources):
  2q 0.027 (Goss22 CZdag Weyl error), 1q 0.003 (qutrit Clifford RB 2.4e-3
  to 5.4e-3), idle_meas 0.0068 (500 ns, Goss22-Q4 coherence, h1_noise_budget),
  idle_reset 0.0034 (250 ns), meas_depol 0, meas_flip 0.025 (3-state
  readout error 0.022-0.031), reset_flip 0.002 (multi-level reset ~1e-3).
  Not modelled (as in SI1000): idle during gate layers, leakage, crosstalk.

Run: module load python/3.11; python h1_dem_readout_schedule.py
Runtime: about 2-4 minutes on one core.
"""

from __future__ import annotations

import dataclasses
import random
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "src"))

from sdim.circuit import Circuit  # noqa: E402

from lrb import code_definitions as defns  # noqa: E402
from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_simulation_profiles import (  # noqa: E402
    CodeSimulationProfileRegistry,
)
from lrb.dem_simulation import (  # noqa: E402
    build_compact_detector_error_model,
    sample_detector_error_model,
    two_qudit_depol_prob_dist,
)

DEPTHS = [0, 2, 6, 10]
N_CLIFF = 4
SHOTS = 20000
SEED = 2026


def parallel_check_block() -> Circuit:
    """All twelve couplings of the repo's check round, then M5..M8 at once."""
    c = Circuit(dimension=3, num_qudits=9)
    c.add_gate("CNOT", 0, 5)
    c.add_gate("CNOT_INV", 1, 5)
    c.add_gate("CNOT_INV", 3, 5)
    c.add_gate("CNOT", 1, 6)
    c.add_gate("CNOT", 2, 6)
    c.add_gate("CNOT_INV", 4, 6)
    c.add_gate("H", 7)
    c.add_gate("CNOT", 7, 0)
    c.add_gate("CNOT", 7, 1)
    c.add_gate("CNOT_INV", 7, 2)
    c.add_gate("H_INV", 7)
    c.add_gate("H", 8)
    c.add_gate("CNOT", 8, 3)
    c.add_gate("CNOT_INV", 8, 1)
    c.add_gate("CNOT_INV", 8, 4)
    c.add_gate("H_INV", 8)
    for a in (5, 6, 7, 8):
        c.add_gate("M", a)
    return c


def code_definitions():
    prof = CodeSimulationProfileRegistry.resolve_code_profile("folded_qutrit")
    seq = prof.code_definition
    par = dataclasses.replace(
        seq, stabilizer_check_blocks=((parallel_check_block, {5, 6, 7, 8}),))
    return {"sequential": seq, "parallel": par}


def classify_locations(circuit):
    """Map operation index of each N1/N2 to a location class."""
    ops = circuit.operations
    out = {}
    for i, op in enumerate(ops):
        if op.gate_name not in ("N1", "N2"):
            continue
        prm = op.params or {}
        scale = float(prm.get("prob_scale", 1.0))
        ch = prm.get("noise_channel", "d")
        if op.gate_name == "N2":
            out[i] = "2q"
            continue
        if ch == "f":
            out[i] = "meas_flip" if scale == 5.0 else "reset_flip"
            continue
        if abs(scale - 0.1) < 1e-12:
            out[i] = "1q"
        elif scale == 2.0:
            j = i + 1
            while ops[j].gate_name in ("N1", "N2"):
                j += 1
            out[i] = "idle_reset" if ops[j].gate_name == "RESET" \
                else "idle_meas"
        elif scale == 1.0:
            out[i] = "meas_depol"
        else:
            raise ValueError((i, op.gate_name, prm))
    return out


def si1000(p):
    return {"2q": p, "1q": 0.1 * p, "idle_meas": 2 * p, "idle_reset": 2 * p,
            "meas_depol": p, "meas_flip": 5 * p, "reset_flip": 2 * p}


HW_A = {"2q": 0.027, "1q": 0.003, "idle_meas": 0.0068, "idle_reset": 0.0034,
        "meas_depol": 0.0, "meas_flip": 0.025, "reset_flip": 0.002}

# Projected (conjecture) operating points with the same coherence-limited
# idle as HW-A: two-qutrit error 1% / 0.5%, three-state readout error 2% / 1%,
# single-qutrit error 1e-3, reset error 1e-3.
HW_B = {"2q": 0.010, "1q": 0.001, "idle_meas": 0.0068, "idle_reset": 0.0034,
        "meas_depol": 0.0, "meas_flip": 0.02, "reset_flip": 0.001}
HW_C = {"2q": 0.005, "1q": 0.001, "idle_meas": 0.0068, "idle_reset": 0.0034,
        "meas_depol": 0.0, "meas_flip": 0.01, "reset_flip": 0.001}

MODELS = {"SI1000 p=0.001": si1000(0.001), "SI1000 p=0.005": si1000(0.005),
          "SI1000 p=0.027": si1000(0.027), "HW-A (p2=0.027)": HW_A,
          "HW-B (p2=0.010)": HW_B, "HW-C (p2=0.005)": HW_C}


def reweighted(model, classes, rates):
    locs = []
    for loc in model.locations:
        r = rates[classes[loc.operation_index]]
        if r <= 0:
            continue
        if loc.gate == "N2":
            params = dict(loc.params)
            params["prob_dist"] = two_qudit_depol_prob_dist(3, r)
            locs.append(dataclasses.replace(loc, probability=r,
                                            params=params))
        else:
            locs.append(dataclasses.replace(loc, probability=r))
    return dataclasses.replace(model, locations=tuple(locs))


def stats(payload, depth):
    det = np.stack([d["data"] for d in payload["detectors"]], axis=1)
    labels = [d["label"] for d in payload["detectors"]]
    log = payload["logicals"][0]["data"]
    term = np.array([lab.startswith(f"lrb_stab_r{depth}_") for lab in labels])
    omega = np.exp(2j * np.pi / 3)
    res = {}
    for pol, acc in (("raw", np.ones(len(log), bool)),
                     ("const1", ~np.any(det[:, term] != 0, axis=1)),
                     ("unif1", ~np.any(det != 0, axis=1))):
        n = int(acc.sum())
        fid = float(np.abs(np.mean(omega ** log[acc]))) if n else float("nan")
        res[pol] = (n / len(log), fid)
    res["det_rate"] = float(np.mean(det != 0))
    return res


def main():
    t0 = time.time()
    random.seed(SEED)
    np.random.seed(SEED)
    cds = code_definitions()
    cliff_seqs = [[random.choice(cds["sequential"].clifford_strings)
                   for _ in range(max(DEPTHS))] for _ in range(N_CLIFF)]
    agg = {}
    for vname, cd in cds.items():
        gen = LRBCircuitGenerator(code_definition=cd, noise_model="si1000")
        for ci, cs in enumerate(cliff_seqs):
            circuits = gen.generate_lrb_clifford_sequence(
                DEPTHS, with_noise=True, clifford_strings=cs)
            for depth, circ in zip(sorted(DEPTHS), circuits):
                gen.update_noise_param(circ, 0.001)
                classes = classify_locations(circ)
                model = build_compact_detector_error_model(circ)
                for mname, rates in MODELS.items():
                    m2 = reweighted(model, classes, rates)
                    payload = sample_detector_error_model(
                        m2, extra_shots=SHOTS)
                    agg.setdefault((mname, vname, depth), []).append(
                        stats(payload, depth))
        print(f"[{time.time() - t0:.0f}s] variant {vname} done", flush=True)

    print(f"\nshots per circuit {SHOTS}, Clifford sequences {N_CLIFF}, "
          f"depths {DEPTHS}; entries: acceptance / fidelity (mean over "
          f"sequences)")
    for mname in MODELS:
        print(f"\n== {mname} ==")
        for depth in DEPTHS:
            line = f"m={depth:2d} "
            for vname in cds:
                rs = agg[(mname, vname, depth)]
                parts = []
                for pol in ("raw", "const1", "unif1"):
                    a = np.mean([r[pol][0] for r in rs])
                    f = np.nanmean([r[pol][1] for r in rs])
                    parts.append(f"{pol} {a:.3f}/{f:.4f}")
                dr = np.mean([r["det_rate"] for r in rs])
                line += f"| {vname}: " + ", ".join(parts) + \
                    f", det-rate {dr:.4f} "
            print(line)
    print(f"\ntotal runtime {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
