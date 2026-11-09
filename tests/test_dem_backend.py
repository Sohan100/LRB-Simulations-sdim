"""The sdim DEM backend matches SDIM's frame sampler through the LRB unpackers.

Circuits are generated with the production generator, written and read back
as CHP files like a real run, and simulated with both backends. The unpacked
stabilizer pass records and logical values must agree in distribution.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

pytest.importorskip("sdim.dem")
from sdim.circuit_io import read_circuit

from lrb.circuit_generator import (
    LRBCircuitGenerator,
    SD6_NOISE_MODEL,
    SI1000_NOISE_MODEL,
)
from lrb.code_simulation_profiles import (
    CodeSimulationProfileRegistry,
    SPLIT_UNIFORM_FOLDED_CODE_NAME,
)
from lrb.dem_simulation import clear_dem_cache
from lrb.lrb_simulation import LRBSimulationPipeline, NoiseModelUtils


SHOTS = 40_000
DEPTH = 2
# Distributions are compared with a two-proportion z-test at this many
# standard errors.
Z_TOLERANCE = 5.0


def generated_circuits(code_name, noise_model, probability, tmp_path):
    """LRB, const=0 and RB circuits at DEPTH, round-tripped through CHP."""
    random.seed(1234)
    profile = CodeSimulationProfileRegistry.resolve_code_profile(code_name)
    generator = LRBCircuitGenerator(
        code_definition=profile.code_definition,
        noise_model=noise_model,
    )
    cliffords = generator.generate_random_clifford_strings(DEPTH)
    families = {
        "lrb": generator.generate_lrb_clifford_sequence(
            [DEPTH], clifford_strings=cliffords)[0],
        "const0": generator.generate_lrb_const0_clifford_sequence(
            [DEPTH], clifford_strings=cliffords)[0],
        "rb": generator.generate_rb_clifford_sequence([DEPTH])[0],
    }
    circuits = {}
    for family, circuit in families.items():
        generator.update_noise_param(circuit, probability)
        path = generator.write_circuit_file(
            circuit, str(tmp_path), f"{family}.chp")
        circuits[family] = read_circuit(path)
        NoiseModelUtils.ensure_noise_params(circuits[family])
    return profile, circuits


def unpacked(profile, family, circuit, backend):
    """Stabilizer pass record and logical values for SHOTS noisy shots."""
    output = LRBSimulationPipeline.simulate_circuit(
        circuit, shots=SHOTS + 1, simulation_backend=backend)
    if family == "rb":
        compact = LRBSimulationPipeline.compact_detector_results_from_sdim_output(
            output)
        reference = compact["references"]["logicals"]["rb_logical"]
        logical = (reference + compact["logicals"]["rb_logical"]) % 3
        return np.ones((SHOTS, 1), dtype=np.int64), logical
    unpack = (profile.unpack_func if family == "lrb"
              else profile.const0_unpack_func)
    checks, logical = unpack(output, DEPTH, SHOTS + 1)
    return np.asarray(checks)[1:], np.asarray(logical)[1:]


def assert_same_rate(successes_a, successes_b, label):
    rate_a = successes_a / SHOTS
    rate_b = successes_b / SHOTS
    pooled = (successes_a + successes_b) / (2 * SHOTS)
    standard_error = np.sqrt(max(pooled * (1 - pooled), 1e-12) * 2 / SHOTS)
    assert abs(rate_a - rate_b) <= Z_TOLERANCE * standard_error + 1e-12, (
        f"{label}: sdim {rate_a:.5f} vs dem {rate_b:.5f}")


@pytest.mark.parametrize("noise_model", [SI1000_NOISE_MODEL, SD6_NOISE_MODEL])
@pytest.mark.parametrize(
    "code_name", ["folded_qutrit", SPLIT_UNIFORM_FOLDED_CODE_NAME])
def test_noiseless_circuits_always_pass(code_name, noise_model, tmp_path):
    clear_dem_cache()
    profile, circuits = generated_circuits(
        code_name, noise_model, 0.0, tmp_path)
    for family, circuit in circuits.items():
        for backend in ("sdim", "dem"):
            checks, logical = unpacked(profile, family, circuit, backend)
            used = checks[:, :DEPTH + 1] if family == "lrb" else checks
            if family == "const0":
                used = checks[:, DEPTH:DEPTH + 1]
            assert np.all(used == 1), (family, backend)
            assert np.all(logical == logical[0]), (family, backend)


@pytest.mark.parametrize("noise_model", [SI1000_NOISE_MODEL, SD6_NOISE_MODEL])
@pytest.mark.parametrize(
    "code_name", ["folded_qutrit", SPLIT_UNIFORM_FOLDED_CODE_NAME])
def test_dem_matches_frame_sampler(code_name, noise_model, tmp_path):
    clear_dem_cache()
    profile, circuits = generated_circuits(
        code_name, noise_model, 0.01, tmp_path)
    _, ideal = generated_circuits(code_name, noise_model, 0.0, tmp_path)
    for family, circuit in circuits.items():
        _, ideal_logical = unpacked(profile, family, ideal[family], "sdim")
        expected = int(ideal_logical[0])
        results = {
            backend: unpacked(profile, family, circuit, backend)
            for backend in ("sdim", "dem")
        }
        (sdim_checks, sdim_logical) = results["sdim"]
        (dem_checks, dem_logical) = results["dem"]
        rounds = (range(DEPTH + 1) if family == "lrb"
                  else [DEPTH] if family == "const0" else [0])
        for check_round in rounds:
            assert_same_rate(
                int(sdim_checks[:, check_round].sum()),
                int(dem_checks[:, check_round].sum()),
                f"{family} pass rate round {check_round}",
            )
        sdim_pass = np.all(sdim_checks[:, list(rounds)] == 1, axis=1)
        dem_pass = np.all(dem_checks[:, list(rounds)] == 1, axis=1)
        assert_same_rate(
            int(np.sum(sdim_pass & (sdim_logical == expected))),
            int(np.sum(dem_pass & (dem_logical == expected))),
            f"{family} accepted and logically correct",
        )
        assert_same_rate(
            int(np.sum(sdim_logical == expected)),
            int(np.sum(dem_logical == expected)),
            f"{family} logically correct",
        )


def test_reference_values_follow_expressions_across_resets():
    from sdim.circuit import Circuit

    circuit = Circuit(2, 3)
    circuit.add_gate("X", 0)
    circuit.add_gate("M", 0)
    circuit.add_gate("RESET", 0)
    circuit.add_gate("X", [0, 0])
    circuit.add_gate("X", 1)
    circuit.add_gate("N1", 1, noise_channel="f", prob=0.0)
    circuit.add_gate("M", [0, 1])
    circuit.add_gate("DETECTOR", expr="rec[1]+-rec[0]", label="d")
    circuit.add_gate("LOGICAL_OBSERVABLE", expr="2*rec[-1]+rec[0]", label="l")
    # Records are 1, 2, 1: the detector is 2 - 1 and the observable 2 + 1.
    expected = {"detectors": {"d": 1}, "logicals": {"l": 0}}

    clear_dem_cache()
    for backend in ("sdim", "dem"):
        output = LRBSimulationPipeline.simulate_circuit(
            circuit, shots=4, simulation_backend=backend)
        compact = LRBSimulationPipeline.compact_detector_results_from_sdim_output(
            output)
        assert compact["references"] == expected, backend
