"""Compact q-ary detector-error-model sampling for annotated LRB circuits."""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    from sdim.program import Program, SimulationOptions, simulate_frame
    _SDIM_DEM_IMPORT_ERROR: Exception | None = None
except Exception as sdim_exc:  # pragma: no cover - environment dependent.
    Program = None
    SimulationOptions = None
    simulate_frame = None
    _SDIM_DEM_IMPORT_ERROR = sdim_exc


GATE_NAMES = {
    17: "N1",
    18: "N2",
    19: "DETECTOR",
    20: "LOGICAL_OBSERVABLE",
}


@dataclass(frozen=True)
class SparseTerms:
    """Sparse detector/logical response to one unit Weyl component."""

    detector: tuple[tuple[int, int], ...]
    logical: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class NoiseLocationResponse:
    """One active noise location in the compact q-ary DEM."""

    noise_location: int
    operation_index: int
    gate: str
    qudit: int
    target: int
    channel: str
    probability: float
    vector_width: int
    params: dict[str, Any]
    component_terms: tuple[SparseTerms, ...]


@dataclass
class CompactDetectorErrorModel:
    """Compiled detector/logical responses for one SDIM circuit."""

    base: int
    detector_labels: tuple[str, ...]
    logical_labels: tuple[str, ...]
    locations: tuple[NoiseLocationResponse, ...]
    all_noise_locations: int
    reference_results: Any
    reference_array: np.ndarray
    ir_array: np.ndarray
    detector_info: Any
    program: Any

    @property
    def num_detectors(self) -> int:
        return len(self.detector_labels)

    @property
    def num_logical_observables(self) -> int:
        return len(self.logical_labels)


_MODEL_CACHE: dict[tuple[int, int], CompactDetectorErrorModel] = {}


def _require_sdim_dem_runtime() -> None:
    if _SDIM_DEM_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "SDIM runtime is unavailable; DEM simulation requires SDIM."
    ) from _SDIM_DEM_IMPORT_ERROR


def simulate_circuit_with_dem(
    circuit,
    shots: int,
    *,
    response_batch_size: int = 256,
    return_metrics: bool = False,
) -> tuple[Any, dict[str, list[dict[str, str | np.ndarray]]]]:
    """
    Return an SDIM-compatible ``(measurements, detector_payload)`` tuple.

    SDIM's frame simulator returns one noiseless reference shot plus
    ``shots - 1`` sampled frame shifts. This DEM backend mirrors that contract:
    the measurement object contains the reference shot, while detector/logical
    arrays contain sampled q-ary shifts for the remaining shots.
    """
    if shots < 1:
        raise ValueError("DEM simulation requires at least one shot.")

    compile_start = time.perf_counter()
    model, cache_hit = cached_compact_detector_error_model(
        circuit,
        response_batch_size=response_batch_size,
        return_cache_hit=True,
    )
    compile_seconds = time.perf_counter() - compile_start

    sample_start = time.perf_counter()
    detector_payload = sample_detector_error_model(
        model,
        extra_shots=shots - 1,
    )
    sample_seconds = time.perf_counter() - sample_start

    simulation_output = (model.reference_results, detector_payload)
    if not return_metrics:
        return simulation_output

    return simulation_output, {
        "dem_compile_seconds": compile_seconds,
        "dem_sample_seconds": sample_seconds,
        "dem_cache_hit": cache_hit,
        "dem_total_noise_locations": model.all_noise_locations,
        "dem_active_noise_locations": len(model.locations),
        "dem_detectors": model.num_detectors,
        "dem_logicals": model.num_logical_observables,
    }


def cached_compact_detector_error_model(
    circuit,
    *,
    response_batch_size: int = 256,
    return_cache_hit: bool = False,
) -> CompactDetectorErrorModel | tuple[CompactDetectorErrorModel, bool]:
    cache_key = (id(circuit), max(1, int(response_batch_size)))
    model = _MODEL_CACHE.get(cache_key)
    cache_hit = model is not None
    if not cache_hit:
        model = build_compact_detector_error_model(
            circuit,
            response_batch_size=response_batch_size,
        )
        _MODEL_CACHE[cache_key] = model
    if return_cache_hit:
        return model, cache_hit
    return model


def build_compact_detector_error_model(
    circuit,
    *,
    response_batch_size: int = 256,
) -> CompactDetectorErrorModel:
    """Compile one annotated SDIM circuit into a compact linear q-ary DEM."""
    _require_sdim_dem_runtime()
    ensure_dem_noise_params(circuit)

    program = Program(circuit)
    program._simulate_tableau(SimulationOptions(shots=1))
    reference_results = program.measurement_results
    reference_array = program._results_to_array(reference_results)
    ir_array, _sampled_noise, detector_info = program._build_ir(
        program.circuits,
        1,
    )

    detector_labels, logical_labels = detector_labels_from_info(detector_info)
    if not detector_labels and not logical_labels:
        raise ValueError(
            "DEM backend requires circuits with DETECTOR or "
            "LOGICAL_OBSERVABLE annotations. Regenerate the circuit inputs "
            "with detector annotations enabled."
        )

    location_specs: list[dict[str, Any]] = []
    component_requests: list[tuple[int, int, int]] = []
    noise_location = 0
    base = int(circuit.dimension)

    for operation_index, instruction in enumerate(circuit.operations):
        if instruction.gate_id not in (17, 18):
            continue

        params = normalized_params(instruction.params)
        vector_width = 2 if instruction.gate_id == 17 else 4
        probability = nonidentity_probability(instruction.gate_id, params, base)
        if probability > 0.0:
            spec_index = len(location_specs)
            component_terms = [SparseTerms((), ()) for _ in range(vector_width)]
            for component_index in range(vector_width):
                component_requests.append(
                    (spec_index, noise_location, component_index)
                )
            location_specs.append(
                {
                    "noise_location": noise_location,
                    "operation_index": operation_index,
                    "gate": GATE_NAMES[instruction.gate_id],
                    "qudit": int(instruction.qudit_index),
                    "target": (
                        -1 if instruction.target_index is None
                        else int(instruction.target_index)
                    ),
                    "channel": params.get(
                        "noise_channel",
                        params.get("channel", "d"),
                    ),
                    "probability": probability,
                    "vector_width": vector_width,
                    "params": params,
                    "component_terms": component_terms,
                }
            )
        noise_location += 1

    if component_requests:
        response_terms = sdim_unit_fault_responses(
            circuit=circuit,
            program=program,
            reference_array=reference_array,
            ir_array=ir_array,
            detector_info=detector_info,
            all_noise_locations=noise_location,
            component_requests=component_requests,
            detector_count=len(detector_labels),
            logical_count=len(logical_labels),
            base=base,
            response_batch_size=response_batch_size,
        )
        for spec_index, _noise_location, component_index in component_requests:
            location_specs[spec_index]["component_terms"][component_index] = (
                response_terms[(spec_index, component_index)]
            )

    locations = tuple(
        NoiseLocationResponse(
            noise_location=spec["noise_location"],
            operation_index=spec["operation_index"],
            gate=spec["gate"],
            qudit=spec["qudit"],
            target=spec["target"],
            channel=spec["channel"],
            probability=spec["probability"],
            vector_width=spec["vector_width"],
            params=spec["params"],
            component_terms=tuple(spec["component_terms"]),
        )
        for spec in location_specs
    )

    return CompactDetectorErrorModel(
        base=base,
        detector_labels=detector_labels,
        logical_labels=logical_labels,
        locations=locations,
        all_noise_locations=noise_location,
        reference_results=reference_results,
        reference_array=reference_array,
        ir_array=ir_array,
        detector_info=detector_info,
        program=program,
    )


def detector_labels_from_info(detector_info) -> tuple[tuple[str, ...], tuple[str, ...]]:
    detector_labels: list[str] = []
    logical_labels: list[str] = []
    for detector_data in detector_info.detector_data:
        label = str(detector_data[1])
        is_logical = bool(detector_data[3])
        if is_logical:
            logical_labels.append(label)
        else:
            detector_labels.append(label)
    return tuple(detector_labels), tuple(logical_labels)


def sdim_unit_fault_responses(
    *,
    circuit,
    program,
    reference_array: np.ndarray,
    ir_array: np.ndarray,
    detector_info,
    all_noise_locations: int,
    component_requests: list[tuple[int, int, int]],
    detector_count: int,
    logical_count: int,
    base: int,
    response_batch_size: int,
) -> dict[tuple[int, int], SparseTerms]:
    """Ask SDIM for detector/logical shifts from unit Weyl faults."""
    batch_size = max(1, int(response_batch_size))
    responses: dict[tuple[int, int], SparseTerms] = {}
    rng_state = np.random.get_state()
    try:
        for start in range(0, len(component_requests), batch_size):
            batch = component_requests[start:start + batch_size]
            batch_shots = len(batch)
            zero_noise = np.zeros(
                (all_noise_locations, batch_shots, 4),
                dtype=np.int64,
            )
            unit_noise = zero_noise.copy()
            for column, (
                _spec_index,
                noise_location,
                component_index,
            ) in enumerate(batch):
                unit_noise[noise_location, column, component_index] = 1

            baseline = simulate_sdim_detector_response(
                circuit=circuit,
                program=program,
                ir_array=ir_array,
                reference_array=reference_array,
                detector_info=detector_info,
                noise_array=zero_noise,
                shots=batch_shots,
            )
            shifted = simulate_sdim_detector_response(
                circuit=circuit,
                program=program,
                ir_array=ir_array,
                reference_array=reference_array,
                detector_info=detector_info,
                noise_array=unit_noise,
                shots=batch_shots,
            )

            for column, (
                spec_index,
                _noise_location,
                component_index,
            ) in enumerate(batch):
                responses[(spec_index, component_index)] = SparseTerms(
                    detector=response_sparse_terms(
                        shifted,
                        baseline,
                        "detectors",
                        column,
                        detector_count,
                        base,
                    ),
                    logical=response_sparse_terms(
                        shifted,
                        baseline,
                        "logicals",
                        column,
                        logical_count,
                        base,
                    ),
                )
    finally:
        np.random.set_state(rng_state)
    return responses


def simulate_sdim_detector_response(
    *,
    circuit,
    program,
    ir_array: np.ndarray,
    reference_array: np.ndarray,
    detector_info,
    noise_array: np.ndarray,
    shots: int,
) -> dict[str, list[dict[str, str | np.ndarray]]]:
    _frame_results, raw_detector_results = simulate_frame(
        ir_array,
        reference_array,
        circuit.num_qudits,
        circuit.dimension,
        shots,
        noise_array,
        detector_info,
    )
    return program._combine_detector_results(detector_info, raw_detector_results)


def response_sparse_terms(
    shifted: dict[str, list[dict[str, str | np.ndarray]]],
    baseline: dict[str, list[dict[str, str | np.ndarray]]],
    block_name: str,
    column: int,
    expected_entries: int,
    base: int,
) -> tuple[tuple[int, int], ...]:
    shifted_entries = shifted.get(block_name, [])
    baseline_entries = baseline.get(block_name, [])
    if len(shifted_entries) != expected_entries:
        raise RuntimeError(
            f"SDIM returned {len(shifted_entries)} {block_name}, "
            f"expected {expected_entries}."
        )
    if len(baseline_entries) != expected_entries:
        raise RuntimeError(
            f"SDIM baseline returned {len(baseline_entries)} {block_name}, "
            f"expected {expected_entries}."
        )

    terms: list[tuple[int, int]] = []
    for index, (shifted_entry, baseline_entry) in enumerate(
        zip(shifted_entries, baseline_entries)
    ):
        if shifted_entry.get("label", "") != baseline_entry.get("label", ""):
            raise RuntimeError(
                f"SDIM {block_name} label mismatch at {index}: "
                f"{shifted_entry.get('label', '')} != "
                f"{baseline_entry.get('label', '')}."
            )
        shifted_data = np.asarray(shifted_entry["data"], dtype=np.int64)
        baseline_data = np.asarray(baseline_entry["data"], dtype=np.int64)
        value = (int(shifted_data[column]) - int(baseline_data[column])) % base
        if value:
            terms.append((index, value))
    return tuple(terms)


def sample_detector_error_model(
    model: CompactDetectorErrorModel,
    *,
    extra_shots: int,
) -> dict[str, list[dict[str, str | np.ndarray]]]:
    if extra_shots < 0:
        raise ValueError("extra_shots must be non-negative.")

    detector_events = np.zeros(
        (extra_shots, model.num_detectors),
        dtype=np.int64,
    )
    logical_events = np.zeros(
        (extra_shots, model.num_logical_observables),
        dtype=np.int64,
    )

    for location in model.locations:
        shot_indices, vectors = sample_location_fault_vectors(
            location,
            model.base,
            extra_shots,
        )
        if len(shot_indices) == 0:
            continue
        apply_location_response(
            detector_events,
            logical_events,
            shot_indices,
            vectors,
            location,
            model.base,
        )

    return {
        "detectors": [
            {"label": label, "data": detector_events[:, index].copy()}
            for index, label in enumerate(model.detector_labels)
        ],
        "logicals": [
            {"label": label, "data": logical_events[:, index].copy()}
            for index, label in enumerate(model.logical_labels)
        ],
    }


def sample_location_fault_vectors(
    location: NoiseLocationResponse,
    base: int,
    shots: int,
) -> tuple[np.ndarray, np.ndarray]:
    if location.gate == "N1":
        applies = np.random.uniform(0.0, 1.0, size=shots) < location.probability
        shot_indices = np.flatnonzero(applies)
        vectors = np.zeros((len(shot_indices), 2), dtype=np.int64)
        if len(shot_indices) == 0:
            return shot_indices, vectors

        if location.channel == "d":
            vectors[:, :2] = sample_nonzero_pauli_powers(
                len(shot_indices),
                base,
                num_powers=2,
            )
        elif location.channel == "f":
            vectors[:, 0] = np.random.randint(1, base, size=len(shot_indices))
        elif location.channel == "p":
            vectors[:, 1] = np.random.randint(1, base, size=len(shot_indices))
        else:
            raise ValueError(f"unsupported N1 DEM channel `{location.channel}`")
        return shot_indices, vectors

    if location.gate == "N2":
        if "prob_dist" in location.params:
            vectors = sample_dense_two_qudit_distribution(
                location.params["prob_dist"],
                base,
                shots,
            )
            nonzero = np.any(vectors != 0, axis=1)
            return np.flatnonzero(nonzero), vectors[nonzero]

        applies = np.random.uniform(0.0, 1.0, size=shots) < location.probability
        shot_indices = np.flatnonzero(applies)
        vectors = np.zeros((len(shot_indices), 4), dtype=np.int64)
        if len(shot_indices) > 0:
            vectors[:, :] = sample_nonzero_pauli_powers(
                len(shot_indices),
                base,
                num_powers=4,
            )
        return shot_indices, vectors

    raise ValueError(f"unsupported DEM noise gate `{location.gate}`")


def apply_location_response(
    detector_events: np.ndarray,
    logical_events: np.ndarray,
    shot_indices: np.ndarray,
    vectors: np.ndarray,
    location: NoiseLocationResponse,
    base: int,
) -> None:
    for component_index, terms in enumerate(location.component_terms):
        component_values = vectors[:, component_index] % base
        nonzero = component_values != 0
        if not np.any(nonzero):
            continue

        selected_shots = shot_indices[nonzero]
        selected_values = component_values[nonzero]
        apply_sparse_terms(
            detector_events,
            selected_shots,
            selected_values,
            terms.detector,
            base,
        )
        apply_sparse_terms(
            logical_events,
            selected_shots,
            selected_values,
            terms.logical,
            base,
        )


def apply_sparse_terms(
    output: np.ndarray,
    shot_indices: np.ndarray,
    values: np.ndarray,
    terms: tuple[tuple[int, int], ...],
    base: int,
) -> None:
    for term_index, coefficient in terms:
        output[shot_indices, term_index] = (
            output[shot_indices, term_index] + values * coefficient
        ) % base


def sample_nonzero_pauli_powers(
    samples: int,
    dimension: int,
    *,
    num_powers: int,
) -> np.ndarray:
    powers = np.random.randint(
        0,
        dimension,
        size=(samples, num_powers),
        dtype=np.int64,
    )
    zero_rows = np.all(powers == 0, axis=1)
    while np.any(zero_rows):
        powers[zero_rows] = np.random.randint(
            0,
            dimension,
            size=(int(np.count_nonzero(zero_rows)), num_powers),
            dtype=np.int64,
        )
        zero_rows = np.all(powers == 0, axis=1)
    return powers


def sample_dense_two_qudit_distribution(
    distribution,
    dimension: int,
    shots: int,
) -> np.ndarray:
    distribution = np.asarray(distribution, dtype=float)
    expected_entries = dimension**4
    if len(distribution) != expected_entries:
        raise ValueError(
            f"Input distribution has length {len(distribution)} instead "
            f"of the required {expected_entries}."
        )

    indices = np.random.choice(a=len(distribution), size=shots, p=distribution)
    noise = np.empty((shots, 4), dtype=np.int64)
    for column in range(3, -1, -1):
        noise[:, column] = indices % dimension
        indices //= dimension
    return noise


def normalized_params(params: dict | None) -> dict[str, Any]:
    normalized = {} if params is None else dict(params)
    prob_dist = normalized.get("prob_dist")
    if isinstance(prob_dist, str):
        normalized["prob_dist"] = ast.literal_eval(prob_dist)
    return normalized


def ensure_dem_noise_params(circuit) -> None:
    """Normalize noise params into the shape expected by SDIM's IR builder."""
    for operation in circuit.operations:
        if operation.gate_id not in (17, 18):
            continue
        if operation.params is None:
            operation.params = {}

        if operation.gate_id == 17:
            if "noise_channel" not in operation.params:
                operation.params["noise_channel"] = operation.params.get(
                    "channel",
                    "d",
                )
            continue

        prob_dist = operation.params.get("prob_dist")
        if isinstance(prob_dist, str):
            prob_dist = ast.literal_eval(prob_dist)
            operation.params["prob_dist"] = prob_dist
        if isinstance(prob_dist, (list, tuple)) and (
                len(prob_dist) == circuit.dimension**4):
            continue

        prob = operation.params.get("prob")
        if prob is None:
            raise ValueError(
                "N2 gate missing 'prob' and 'prob_dist' params; cannot "
                "compile DEM."
            )
        operation.params["prob_dist"] = two_qudit_depol_prob_dist(
            circuit.dimension,
            float(prob),
        )


def two_qudit_depol_prob_dist(dimension: int, prob: float) -> list[float]:
    if not 0.0 <= prob <= 1.0:
        raise ValueError(f"Invalid depolarizing prob {prob}")
    n = dimension**4
    if n <= 1:
        return [1.0]
    tail = prob / (n - 1)
    distribution = [tail] * n
    distribution[0] = 1.0 - prob
    return distribution


def nonidentity_probability(gate_id: int, params: dict[str, Any], base: int) -> float:
    if gate_id == 17:
        return float(params.get("prob", 0.0))
    if gate_id == 18:
        if "prob_dist" in params:
            distribution = np.asarray(params["prob_dist"], dtype=float)
            return float(
                sum(
                    probability
                    for index, probability in enumerate(distribution)
                    if probability > 0.0
                    and any(index_to_digits(index, base, 4))
                )
            )
        return float(params.get("prob", 0.0))
    return 0.0


def index_to_digits(index: int, base: int, width: int) -> list[int]:
    digits = [0] * width
    for column in range(width - 1, -1, -1):
        digits[column] = index % base
        index //= base
    return digits
