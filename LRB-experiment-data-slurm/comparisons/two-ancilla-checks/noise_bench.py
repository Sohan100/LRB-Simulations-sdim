"""Noisy memory benchmark of one check round, with the production noise.

|+L> (noiseless), optionally logical H; R check rounds, each scheduled and
given SI1000 or SD6 noise by LRBCircuitGenerator.apply_circuit_level_noise
exactly as in the LRB runs; optionally logical H^-1; noiseless X readout of the
data. A shot is accepted if every ancilla readout is 0 and G3, G4 from the
final data are 0. Reported: acceptance and the logical error rate among
accepted shots (X basis on |+L>, and with the H sandwich).

Usage: noise_bench.py MODEL P [ROUNDS] [SHOTS]
"""
import pickle
import sys
from pathlib import Path

from sdim.circuit import Circuit
from sdim.program import Program
from lrb.circuit_generator import LRBCircuitGenerator
from lrb.code_simulation_profiles import CodeSimulationProfileRegistry as Reg

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import protocol_search as PS  # noqa: E402


def production_round(code_name, model):
    prof = Reg.resolve_code_profile(code_name)
    gen = LRBCircuitGenerator(code_definition=prof.code_definition, noise_model=model)
    return gen._check_round_circuit(), prof


def ops_round(ops, nq):
    return PS.to_circuit(ops, nq)


def collapse_wires(circuit, gen):
    """(gate, wire) for every M / RESET in the ideal round, in order."""
    out = []
    for op in circuit.operations:
        if op.gate_name in ("M", "RESET"):
            out.append((op.gate_name, int(op.qudit_index)))
    return out


def memory(code, ideal_round, model, p, rounds, sandwich, shots):
    gen = LRBCircuitGenerator(noise_model=model)
    c = code.logical_plus_initial_state()
    if sandwich:
        c = c + code.single_qudit_logical_gate_circuit("H")
    for _ in range(rounds):
        c = c + gen.apply_circuit_level_noise(ideal_round)
    if sandwich:
        c = c + code.single_qudit_logical_gate_circuit("H_INV")
    for q in range(5):
        c.add_gate("H_INV", q); c.add_gate("M", q)
    gen.update_noise_param(c, p)
    # result index of every ancilla readout and final data readout
    count, reads, final = {}, [], []
    for _ in range(rounds):
        for gate, w in collapse_wires(ideal_round, gen):
            k = count.get(w, 0)
            if gate == "M":
                reads.append((w, k))
            count[w] = k + 1
    if sandwich or True:
        pass
    # prep / logical-gate circuits contain no M or RESET on these wires
    for q in range(5):
        final.append((q, count.get(q, 0)))
    m = Program(c).simulate(shots=shots)[0]
    acc = err = 0
    for s in range(shots):
        if any(int(m[w][k][s].measurement_value) for w, k in reads):
            continue
        x = [int(m[w][k][s].measurement_value) for w, k in final]
        if (x[0] + x[1] - x[2]) % 3 or (-x[1] + x[3] - x[4]) % 3:
            continue
        acc += 1
        err += (x[0] + x[3]) % 3 != 0
    return acc, err


def schemes(model):
    best = pickle.load(open(HERE / "protocol_search_best.pkl", "rb"))
    one, prof1 = production_round("folded_qutrit", model)
    rel, prof2 = production_round("folded_qutrit_split_unif", model)
    from lrb.code_definitions import (FoldedQutritDetectionCode as C9,
                                      FoldedQutritSplitAncillaUniformDetectionCode as C11)
    return {
        "one ancilla (current)": (C9, one),
        "coherent relay (current)": (C11, rel),
        "data relay, mid-round readout": (C9, ops_round(best["mid"][0][3], 9)),
        "data relay, measure once": (C9, ops_round(best["once"][0][3], 9)),
    }


if __name__ == "__main__":
    model, p = sys.argv[1], float(sys.argv[2])
    rounds = int(sys.argv[3]) if len(sys.argv) > 3 else 3
    shots = int(sys.argv[4]) if len(sys.argv) > 4 else 20000
    for name, (code, ideal) in schemes(model).items():
        cells = []
        for sandwich in (False, True):
            acc, err = memory(code, ideal, model, p, rounds, sandwich, shots)
            cells.append(f"acc {acc / shots:.4f} err {err / max(acc, 1):.2e} ({err})")
        print(f"{model} p={p} R={rounds}  {name:32s}  X: {cells[0]}   Z: {cells[1]}", flush=True)
