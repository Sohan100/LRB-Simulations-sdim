"""Detector-error-model sampling for annotated LRB circuits via ``sdim.dem``.

SDIM 1.4.0 ships an exact compact q-ary detector error model,
``sdim.dem.DetectorErrorModel``. Each N1/N2 noise gate becomes one error
mechanism whose generators are the detector/logical responses of the gate's
unit X and Z faults, so depolarizing channels are represented exactly rather
than as independent per-Pauli events. This module wraps that model in the
``(measurements, detector_payload)`` contract used by the SDIM frame sampler,
so the LRB post-processing path is unchanged.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from sdim.circuit import Circuit
    from sdim.dem import DetectorErrorModel
    from sdim.program import Program
    _SDIM_DEM_IMPORT_ERROR: Exception | None = None
except Exception as sdim_exc:  # pragma: no cover - environment dependent.
    Circuit = None
    DetectorErrorModel = None
    Program = None
    _SDIM_DEM_IMPORT_ERROR = sdim_exc

from .detector_events import reference_event_values


NOISE_GATE_IDS = (17, 18)


@dataclass
class CompiledCircuitDem:
    """One circuit's DEM, its compiled sampler, and its reference shot."""

    circuit: Any
    dem: Any
    sampler: Any
    reference_results: Any
    reference_values: dict[str, list[int]]
    total_noise_locations: int

    @property
    def num_detectors(self) -> int:
        return int(self.dem.num_detectors)

    @property
    def num_logical_observables(self) -> int:
        return int(self.dem.num_observables)


# Keyed by id(circuit); the entry keeps the circuit alive, so the id cannot be
# reused by another circuit while the entry exists.
_MODEL_CACHE: dict[int, CompiledCircuitDem] = {}


def _require_sdim_dem_runtime() -> None:
    if _SDIM_DEM_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "sdim.dem is unavailable; the DEM backend requires sdim>=1.4.0."
    ) from _SDIM_DEM_IMPORT_ERROR


def clear_dem_cache() -> None:
    """Drop every cached model and sampler."""
    _MODEL_CACHE.clear()


def simulate_circuit_with_dem(
    circuit,
    shots: int,
    *,
    return_metrics: bool = False,
) -> tuple[Any, dict[str, list[dict[str, Any]]]]:
    """
    Return an SDIM-compatible ``(measurements, detector_payload)`` tuple.

    SDIM's frame simulator returns one noiseless reference shot plus
    ``shots - 1`` sampled frame shifts. This DEM backend mirrors that contract:
    the measurement object contains the reference shot, while detector/logical
    arrays contain sampled q-ary shifts for the remaining shots. Each payload
    entry also carries the event's absolute value on the reference shot under
    ``"reference"``.

    The model and its sampler are compiled once per circuit object and reused
    by later calls, which continue the sampler's random stream.
    """
    if shots < 1:
        raise ValueError("DEM simulation requires at least one shot.")

    compile_start = time.perf_counter()
    model, cache_hit = cached_circuit_dem(circuit, return_cache_hit=True)
    compile_seconds = time.perf_counter() - compile_start

    sample_start = time.perf_counter()
    detector_payload = sample_circuit_dem(model, extra_shots=shots - 1)
    sample_seconds = time.perf_counter() - sample_start

    simulation_output = (model.reference_results, detector_payload)
    if not return_metrics:
        return simulation_output

    return simulation_output, {
        "dem_compile_seconds": compile_seconds,
        "dem_sample_seconds": sample_seconds,
        "dem_cache_hit": cache_hit,
        "dem_total_noise_locations": model.total_noise_locations,
        "dem_active_noise_locations": len(model.dem.mechanisms),
        "dem_detectors": model.num_detectors,
        "dem_logicals": model.num_logical_observables,
    }


def cached_circuit_dem(
    circuit,
    *,
    return_cache_hit: bool = False,
) -> CompiledCircuitDem | tuple[CompiledCircuitDem, bool]:
    model = _MODEL_CACHE.get(id(circuit))
    cache_hit = model is not None and model.circuit is circuit
    if not cache_hit:
        model = build_circuit_dem(circuit)
        _MODEL_CACHE[id(circuit)] = model
    if return_cache_hit:
        return model, cache_hit
    return model


def noiseless_copy(circuit):
    """Return a circuit with the same operations minus N1/N2 noise gates."""
    _require_sdim_dem_runtime()
    copy = Circuit(circuit.num_qudits, circuit.dimension)
    copy.operations = [
        operation for operation in circuit.operations
        if operation.gate_id not in NOISE_GATE_IDS
    ]
    return copy


def reference_shot(circuit):
    """
    Simulate the noiseless reference shot of ``circuit``.

    Returns:
        tuple: The SDIM measurement tensor ``[qudit][round][shot]`` holding
        one shot, and the reference values of every detector and logical
        observable.
    """
    program = Program(noiseless_copy(circuit))
    program.simulate(shots=1)
    reference_results = program.measurement_results
    return reference_results, reference_event_values(
        circuit, reference_results)


def build_circuit_dem(circuit, *, seed=None) -> CompiledCircuitDem:
    """Compile one annotated SDIM circuit into an sdim DEM and sampler."""
    _require_sdim_dem_runtime()
    reference_results, reference_values = reference_shot(circuit)
    dem = DetectorErrorModel.from_circuit(circuit)
    if dem.num_detectors == 0 and dem.num_observables == 0:
        raise ValueError(
            "DEM backend requires circuits with DETECTOR or "
            "LOGICAL_OBSERVABLE annotations. Regenerate the circuit inputs "
            "with detector annotations enabled."
        )
    return CompiledCircuitDem(
        circuit=circuit,
        dem=dem,
        sampler=dem.compile_sampler(seed=seed),
        reference_results=reference_results,
        reference_values=reference_values,
        total_noise_locations=sum(
            1 for operation in circuit.operations
            if operation.gate_id in NOISE_GATE_IDS
        ),
    )


def sample_circuit_dem(
    model: CompiledCircuitDem,
    *,
    extra_shots: int,
) -> dict[str, list[dict[str, Any]]]:
    if extra_shots < 0:
        raise ValueError("extra_shots must be non-negative.")

    detector_events, logical_events = model.sampler.sample(extra_shots)
    return {
        "detectors": event_entries(
            model.dem.detector_labels,
            detector_events,
            model.reference_values["detectors"],
        ),
        "logicals": event_entries(
            model.dem.observable_labels,
            logical_events,
            model.reference_values["logicals"],
        ),
    }


def event_entries(
    labels,
    events: np.ndarray,
    references: list[int],
) -> list[dict[str, Any]]:
    if len(labels) != len(references):
        raise RuntimeError(
            f"DEM has {len(labels)} events but the circuit's reference shot "
            f"evaluated {len(references)}."
        )
    return [
        {
            "label": label,
            "data": np.ascontiguousarray(events[:, index]),
            "reference": reference,
        }
        for index, (label, reference) in enumerate(zip(labels, references))
    ]
