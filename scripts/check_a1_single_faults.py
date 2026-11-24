"""Single-fault enumeration of the A1 check round as the generator builds it.

For the qutrit and qubit A1 profiles, every single Pauli fault is inserted
after every operation of one check round (after each gate and reset, and both
before and after each readout), in a noiseless circuit:

    encoder, [logical H], round (with the fault), round, [logical H_INV],
    X-basis readout of the five data qudits.

A fault is silent when no ancilla readout of either round is nonzero, both
terminal X stabilizers are satisfied, and the logical X value is wrong. The
run with the logical-H sandwich turns X-type logical errors into detectable
Z-type ones. Every two-qudit fault takes all d^4 - 1 two-qudit Paulis.

    PYTHONNOUSERSITE=1 PYTHONPATH=src .venv-sdim140/bin/python \
        scripts/check_a1_single_faults.py [--processes 16]
"""

from __future__ import annotations

import argparse
import itertools
import sys
from multiprocessing import Pool
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from sdim.circuit import Circuit  # noqa: E402
from sdim.program import Program  # noqa: E402

from lrb.circuit_generator import LRBCircuitGenerator  # noqa: E402
from lrb.code_simulation_profiles import (  # noqa: E402
    A1_SQUARE_FOLDED_CODE_NAME,
    A1_SQUARE_FOLDED_QUBIT_CODE_NAME,
    CodeSimulationProfileRegistry,
)

ANCILLAS = (5, 6)


def setup(code_name):
    definition = CodeSimulationProfileRegistry.resolve_code_profile(
        code_name).code_definition
    generator = LRBCircuitGenerator(code_definition=definition,
                                    noise_model="si1000")
    return definition, generator._check_round_circuit()


def paulis(d, n):
    return [p for p in itertools.product(range(d), repeat=2 * n) if any(p)]


def fault_sites(round_circuit, d):
    """(op index, 'after'|'before', qudits, Pauli exponents)."""
    sites = []
    for i, op in enumerate(round_circuit.operations):
        wires = tuple(LRBCircuitGenerator._operation_qudits(op))
        if op.gate_name == "M":
            for k in range(1, d):
                sites.append((i, "before", wires, (k, 0)))
            for p in paulis(d, 1):
                sites.append((i, "after", wires, p))
        else:
            for p in paulis(d, len(wires)):
                sites.append((i, "after", wires, p))
    return sites


def add_pauli(c, wires, exps):
    for j, q in enumerate(wires):
        x, z = exps[2 * j], exps[2 * j + 1]
        for _ in range(x):
            c.add_gate("X", q)
        for _ in range(z):
            c.add_gate("Z", q)


def build(code_name, sandwich, site):
    definition, round_circuit = setup(code_name)
    d = definition.dimension
    c = definition.logical_plus_initial_state()
    if sandwich:
        c = c + definition.logical_gate_circuit("H")
    faulty = Circuit(dimension=d, num_qudits=definition.encoded_num_qudits)
    generator = LRBCircuitGenerator(noise_model="si1000")
    for i, op in enumerate(round_circuit.operations):
        if site is not None and site[0] == i and site[1] == "before":
            add_pauli(faulty, site[2], site[3])
        generator._append_operation_copy(faulty, op)
        if site is not None and site[0] == i and site[1] == "after":
            add_pauli(faulty, site[2], site[3])
    c = c + faulty + round_circuit
    if sandwich:
        c = c + definition.logical_gate_circuit("H_INV")
    for q in range(5):
        c.add_gate("H_INV", q)
        c.add_gate("M", q)
    return c, d


def classify(args):
    code_name, sandwich, site = args
    c, d = build(code_name, sandwich, site)
    m = Program(c).simulate(shots=2)[0]
    val = lambda w, k, s: int(m[w][k][s].measurement_value) % d
    out = []
    for s in range(2):
        anc = [val(w, k, s) for w in ANCILLAS for k in range(len(m[w]))]
        x = [val(q, 0, s) for q in range(5)]
        chk = any(anc)
        dat = ((x[0] + x[1] - x[2]) % d or (-x[1] + x[3] - x[4]) % d)
        lg = (x[0] + x[3]) % d
        out.append(bool(lg and not chk and not dat))
    assert out[0] == out[1], (site, out)
    return site, out[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processes", type=int, default=16)
    args = parser.parse_args()
    with Pool(args.processes) as pool:
        for code_name in (A1_SQUARE_FOLDED_CODE_NAME,
                          A1_SQUARE_FOLDED_QUBIT_CODE_NAME):
            definition, round_circuit = setup(code_name)
            d = definition.dimension
            for sandwich in (False, True):
                ok, _ = classify((code_name, sandwich, None))
                assert not ok
                sites = fault_sites(round_circuit, d)
                results = pool.map(classify,
                                   [(code_name, sandwich, s) for s in sites],
                                   chunksize=8)
                silent = [s for s, bad in results if bad]
                label = " (logical-H sandwich)" if sandwich else ""
                print(f"{code_name}{label}: {len(sites)} single faults, "
                      f"{len(silent)} silent logical"
                      + (f"; e.g. {silent[:3]}" if silent else ""),
                      flush=True)


if __name__ == "__main__":
    main()
