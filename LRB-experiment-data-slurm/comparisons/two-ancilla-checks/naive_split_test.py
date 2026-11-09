"""Noiseless test: split G1 and G4 over two ancillas measured separately
(readouts added classically) vs the simulated coherent relay.

Wires: data 0-4; G1 parts on 5 (q0,q1 = site N) and 9 (q3 = site W);
G2 on 6 (site E); G3 on 7 (site N); G4 parts on 8 (q1,q4 = site E) and
10 (q3 = site S). Two check rounds, then the terminal logical-X readout.
"""
from collections import Counter

from sdim.circuit import Circuit
from sdim.program import Program
from lrb.code_definitions import FoldedQutritSplitAncillaUniformDetectionCode as Code

N = Code.num_qudits


def naive_g1():
    c = Circuit(dimension=3, num_qudits=N)
    c.add_gate("CNOT", 0, 5); c.add_gate("CNOT_INV", 1, 5); c.add_gate("M", 5)
    c.add_gate("CNOT_INV", 3, 9); c.add_gate("M", 9)
    return c


def naive_g4():
    c = Circuit(dimension=3, num_qudits=N)
    c.add_gate("H", 8); c.add_gate("CNOT_INV", 8, 1); c.add_gate("CNOT_INV", 8, 4)
    c.add_gate("H_INV", 8); c.add_gate("M", 8)
    c.add_gate("H", 10); c.add_gate("CNOT", 10, 3); c.add_gate("H_INV", 10); c.add_gate("M", 10)
    return c


def pair_g1():
    c = Circuit(dimension=3, num_qudits=N)
    c.add_gate("H", 5); c.add_gate("CNOT", 5, 9)            # entangle the pair first
    c.add_gate("CZ", 5, 0); c.add_gate("CZ_INV", 5, 1); c.add_gate("CZ_INV", 9, 3)
    c.add_gate("H_INV", 5); c.add_gate("H_INV", 9); c.add_gate("M", 5); c.add_gate("M", 9)
    return c


def pair_g4():
    c = Circuit(dimension=3, num_qudits=N)
    c.add_gate("H", 8); c.add_gate("CNOT", 8, 10)           # entangle the pair first
    c.add_gate("CNOT_INV", 8, 1); c.add_gate("CNOT_INV", 8, 4); c.add_gate("CNOT", 10, 3)
    c.add_gate("H_INV", 8); c.add_gate("H_INV", 10); c.add_gate("M", 8); c.add_gate("M", 10)
    return c


SCHEMES = {
    "simulated coherent relay": [
        Code.z_boundary_split_stabilizer_measurement_ancillae_circuit,
        Code.z_middle_stabilizer_measurement_ancillae_circuit,
        Code.x_middle_stabilizer_measurement_ancillae_circuit,
        Code.x_boundary_split_stabilizer_measurement_ancillae_circuit],
    "separate readout, added": [
        naive_g1,
        Code.z_middle_stabilizer_measurement_ancillae_circuit,
        Code.x_middle_stabilizer_measurement_ancillae_circuit,
        naive_g4],
    "separate readout, G4 first": [
        naive_g4,
        Code.z_middle_stabilizer_measurement_ancillae_circuit,
        Code.x_middle_stabilizer_measurement_ancillae_circuit,
        naive_g1],
    "entangled pair, added": [
        pair_g1,
        Code.z_middle_stabilizer_measurement_ancillae_circuit,
        Code.x_middle_stabilizer_measurement_ancillae_circuit,
        pair_g4],
}
SHOTS = 3000
ERROR = None
for name, blocks in SCHEMES.items():
    circuit = Code.logical_plus_initial_state()
    if ERROR:
        circuit.add_gate(ERROR[1], ERROR[0])
    for _ in range(2):
        for block in blocks:
            circuit = circuit + block()
        circuit = circuit + Code.reset_measurement_wires()
    circuit = circuit + Code.terminal_logical_x_measurement_circuit()
    m = Program(circuit).simulate(shots=SHOTS)
    if isinstance(m, tuple):
        m = m[0]
    v = lambda w, k, s: int(m[w][2 * k if w >= 5 else k][s].measurement_value)  # RESET also records a result
    print(f"\n== {name} ==")
    for rnd in range(2):
        g1 = Counter((v(5, rnd, s) + v(9, rnd, s)) % 3 for s in range(SHOTS))
        g4 = Counter((v(8, rnd, s) + v(10, rnd, s)) % 3 for s in range(SHOTS))
        g2 = Counter(v(6, rnd, s) for s in range(SHOTS))
        g3 = Counter(v(7, rnd, s) for s in range(SHOTS))
        print(f" round {rnd + 1}: G1 {dict(g1)}  G2 {dict(g2)}  G3 {dict(g3)}  G4 {dict(g4)}")
    passed = [s for s in range(SHOTS) if all(
        (v(5, r, s) + v(9, r, s)) % 3 == 0 and v(6, r, s) == 0 and v(7, r, s) == 0
        and (v(8, r, s) + v(10, r, s)) % 3 == 0 for r in range(2))]
    xbar = Counter((v(0, 0, s) + v(3, 0, s)) % 3 for s in passed)
    print(f" pass both rounds: {len(passed)}/{SHOTS};  logical X (x0+x3) among passed: {dict(xbar)}")

# Single errors before the first round, entangled-pair scheme only.
for ERROR in [(3, "X"), (3, "Z"), (1, "X"), (1, "Z"), (0, "X"), (4, "Z")]:
    blocks = SCHEMES["entangled pair, added"]
    circuit = Code.logical_plus_initial_state()
    circuit.add_gate(ERROR[1], ERROR[0])
    for block in blocks:
        circuit = circuit + block()
    m = Program(circuit).simulate(shots=300)[0]
    v = lambda w, s: int(m[w][0][s].measurement_value)
    sums = {(v(5, s) + v(9, s)) % 3 for s in range(300)}, {v(6, s) for s in range(300)}, \
        {v(7, s) for s in range(300)}, {(v(8, s) + v(10, s)) % 3 for s in range(300)}
    single = {v(5, s) for s in range(300)}
    print(f"entangled pair, {ERROR[1]} on q{ERROR[0]}: G1 {sums[0]} G2 {sums[1]} G3 {sums[2]} G4 {sums[3]}; a1 alone {single}")
