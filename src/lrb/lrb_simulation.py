"""Simulation and post-processing utilities for logical RB workflows."""

from __future__ import annotations

import ast
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
try:
    from sdim.circuit import Circuit
    from sdim.circuit_io import read_circuit
    from sdim.program import Program
    _SDIM_IMPORT_ERROR: Exception | None = None
except Exception as sdim_exc:  # pragma: no cover - environment dependent.
    # Keep plotting/stat utilities importable even when SDIM runtime is absent.
    Circuit = Any
    read_circuit = None
    Program = None
    _SDIM_IMPORT_ERROR = sdim_exc

from .experiment_setup import ExperimentSetupManager
from .detector_events import (
    LRB_CONST0_LOGICAL_OBSERVABLE_LABEL,
    LRB_LOGICAL_OBSERVABLE_LABEL,
    RB_LOGICAL_OBSERVABLE_LABEL,
    lrb_const0_stabilizer_detector_label,
    lrb_stabilizer_detector_label,
)


NORMAL_RB_SHOTS = 10000
SIMULATION_BACKEND_ENV = "LRB_SIMULATION_BACKEND"
DEM_RESPONSE_BATCH_SIZE_ENV = "LRB_DEM_RESPONSE_BATCH_SIZE"
SUPPORTED_SIMULATION_BACKENDS = ("sdim", "dem")
TIMING_METRIC_FILENAME = "timing_metrics.csv"
TIMING_METRIC_FIELDS = (
    "timestamp_utc",
    "backend",
    "phase",
    "event",
    "probability_index",
    "probability",
    "batch_index",
    "batch_shots",
    "shots_remaining_before",
    "shots_remaining_after",
    "clifford_index",
    "depth_index",
    "depth",
    "shots",
    "circuit_dimension",
    "circuit_qudits",
    "circuit_operations",
    "simulator_seconds",
    "dem_compile_seconds",
    "dem_sample_seconds",
    "dem_cache_hit",
    "dem_total_noise_locations",
    "dem_active_noise_locations",
    "dem_detectors",
    "dem_logicals",
    "unpack_seconds",
    "record_seconds",
    "process_seconds",
    "load_seconds",
    "write_seconds",
    "total_seconds",
    "notes",
)


def _resolve_simulation_backend(
        simulation_backend: str | None = None) -> str:
    """Return the selected circuit-sampling backend."""
    backend = simulation_backend or os.environ.get(SIMULATION_BACKEND_ENV)
    if backend is None:
        use_dem = os.environ.get("LRB_USE_DEM", "").strip().lower()
        backend = "dem" if use_dem in {"1", "true", "yes", "on"} else "sdim"

    backend = backend.strip().lower()
    if backend not in SUPPORTED_SIMULATION_BACKENDS:
        supported = ", ".join(SUPPORTED_SIMULATION_BACKENDS)
        raise ValueError(
            f"Unsupported simulation backend '{backend}'. "
            f"Supported backends are: {supported}."
        )
    return backend


def _dem_response_batch_size() -> int:
    value = int(os.environ.get(DEM_RESPONSE_BATCH_SIZE_ENV, "256"))
    if value < 1:
        raise ValueError(f"{DEM_RESPONSE_BATCH_SIZE_ENV} must be at least 1.")
    return value


def _format_timing_metric_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, float):
        return f"{value:.9f}"
    if isinstance(value, np.generic):
        return value.item()
    return value


def _append_timing_metric(
    partial_progress_folder: str | None,
    metrics: dict[str, Any],
) -> None:
    if not partial_progress_folder:
        return

    os.makedirs(partial_progress_folder, exist_ok=True)
    metrics_path = os.path.join(
        partial_progress_folder,
        TIMING_METRIC_FILENAME,
    )
    row = {field: "" for field in TIMING_METRIC_FIELDS}
    row["timestamp_utc"] = time.strftime(
        "%Y-%m-%dT%H:%M:%SZ",
        time.gmtime(),
    )
    for field in TIMING_METRIC_FIELDS:
        if field in metrics:
            row[field] = _format_timing_metric_value(metrics[field])

    should_write_header = (
        not os.path.exists(metrics_path)
        or os.path.getsize(metrics_path) == 0
    )
    with open(metrics_path, "a", newline="") as metrics_file:
        writer = csv.DictWriter(metrics_file, fieldnames=TIMING_METRIC_FIELDS)
        if should_write_header:
            writer.writeheader()
        writer.writerow(row)


def atomic_write_text(filename: str, value: str) -> None:
    """Durably replace a small text file without exposing a partial write."""
    parent = os.path.dirname(os.path.abspath(filename))
    os.makedirs(parent, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=parent,
        prefix=f".{os.path.basename(filename)}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w") as writer:
            writer.write(value)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, filename)
        _fsync_directory(parent)
    except BaseException:
        if os.path.exists(temporary):
            os.unlink(temporary)
        raise


def _fsync_directory(directory: str) -> None:
    """Flush directory-entry changes when the platform supports it."""
    descriptor = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_tree(root: str) -> None:
    """Flush every file and directory in a staged checkpoint tree."""
    directories = []
    for directory, child_directories, filenames in os.walk(root):
        directories.append(directory)
        for filename in filenames:
            path = os.path.join(directory, filename)
            if os.path.islink(path):
                continue
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for child_directory in child_directories:
            child_path = os.path.join(directory, child_directory)
            if os.path.islink(child_path):
                raise RuntimeError(
                    f"Checkpoint staging tree contains a symlink: "
                    f"{child_path}"
                )
    for directory in reversed(directories):
        _fsync_directory(directory)


def _critical_runtime_source_fingerprint() -> str:
    """Hash simulation sources whose drift would invalidate a checkpoint."""
    module_root = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(module_root))
    relative_paths = (
        "src/lrb/lrb_simulation.py",
        "src/lrb/dem_simulation.py",
        "src/lrb/detector_events.py",
        "src/lrb/code_definitions.py",
        "src/lrb/code_simulation_profiles.py",
        "scripts/run_lrb_experiment.py",
    )
    digest = hashlib.sha256()
    for relative_path in relative_paths:
        absolute_path = os.path.join(project_root, relative_path)
        digest.update(relative_path.encode("utf-8"))
        with open(absolute_path, "rb") as source_file:
            digest.update(source_file.read())
    return digest.hexdigest()


def _installed_distribution_version(distribution: str) -> str:
    """Return a stable package version for checkpoint compatibility checks."""
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _checkpoint_test_failpoint(name: str) -> None:
    """Raise only when a private test failpoint is explicitly selected."""
    if os.environ.get("_LRB_TEST_CHECKPOINT_FAILPOINT") == name:
        raise RuntimeError(f"Injected checkpoint failure at {name}.")


def _circuit_input_fingerprint(
    circuit_folders: Sequence[tuple[str, str]],
    *,
    num_cliff_seq: int,
    probability_index: int,
    num_depths: int,
) -> str:
    """Hash every circuit consumed by one probability-index worker."""
    digest = hashlib.sha256()
    for family, folder in circuit_folders:
        folder = os.path.abspath(folder)
        for clifford_index in range(num_cliff_seq):
            for depth_index in range(num_depths):
                relative_path = os.path.join(
                    str(clifford_index),
                    str(probability_index),
                    f"{depth_index}.chp",
                )
                circuit_path = os.path.join(folder, relative_path)
                if not os.path.isfile(circuit_path):
                    raise RuntimeError(
                        f"Missing expected {family} circuit: {circuit_path}"
                    )
                if os.path.getsize(circuit_path) == 0:
                    raise RuntimeError(
                        f"Empty expected {family} circuit: {circuit_path}"
                    )
                digest.update(family.encode("utf-8"))
                digest.update(relative_path.encode("utf-8"))
                with open(circuit_path, "rb") as circuit_file:
                    for chunk in iter(
                        lambda: circuit_file.read(1024 * 1024),
                        b"",
                    ):
                        digest.update(chunk)
    return digest.hexdigest()


class LRBBatchCheckpointStore:
    """Copy-on-write, atomically published LRB batch checkpoints."""

    GENERATIONS_DIRECTORY = ".checkpoint_generations"
    CURRENT_LINK = ".checkpoint_current"
    MANIFEST_FILENAME = "manifest.json"
    SHOTS_FILENAME = "shots_processed.txt"
    COUNT_DIRECTORIES = ("const_check_data", "unif_check_data")

    def __init__(
        self,
        progress_folder: str,
        *,
        num_shots: int,
        batch_size: int,
        num_cliff_seq: int,
        depths: Sequence[int],
        const_checks: Sequence[int],
        unif_checks: Sequence[int],
        dimension: int,
        probability_index: int,
        probability: float,
        backend: str,
        input_fingerprint: str,
        runtime_profile: str,
        filter_trivial_shots: bool,
    ) -> None:
        self.progress_folder = os.path.abspath(progress_folder)
        self.generations_directory = os.path.join(
            self.progress_folder,
            self.GENERATIONS_DIRECTORY,
        )
        self.current_link = os.path.join(
            self.progress_folder,
            self.CURRENT_LINK,
        )
        self.num_shots = int(num_shots)
        self.batch_size = int(batch_size)
        self.num_cliff_seq = int(num_cliff_seq)
        self.depths = tuple(int(depth) for depth in depths)
        self.const_checks = tuple(int(check) for check in const_checks)
        self.unif_checks = tuple(int(check) for check in unif_checks)
        self.dimension = int(dimension)
        self.probability_index = int(probability_index)
        self.probability = float(probability)
        self.backend = str(backend)
        self.input_fingerprint = str(input_fingerprint)
        self.runtime_profile = str(runtime_profile)
        self.filter_trivial_shots = bool(filter_trivial_shots)
        self._active_transaction: str | None = None

        if self.num_shots < 1:
            raise ValueError("Checkpoint shot count must be positive.")
        if self.batch_size < 1:
            raise ValueError("Checkpoint batch size must be positive.")
        if self.num_cliff_seq < 1 or not self.depths:
            raise ValueError("Checkpoint circuit dimensions must be positive.")

        self.configuration = {
            "schema_version": 1,
            "num_shots": self.num_shots,
            "batch_size": self.batch_size,
            "num_cliff_seq": self.num_cliff_seq,
            "depths": list(self.depths),
            "const_checks": list(self.const_checks),
            "unif_checks": list(self.unif_checks),
            "dimension": self.dimension,
            "probability_index": self.probability_index,
            "probability": self.probability,
            "backend": self.backend,
            "input_fingerprint": self.input_fingerprint,
            "runtime_profile": self.runtime_profile,
            "filter_trivial_shots": self.filter_trivial_shots,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "sdim_version": _installed_distribution_version("sdim"),
            "source_fingerprint": _critical_runtime_source_fingerprint(),
        }
        encoded_configuration = json.dumps(
            self.configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        self.configuration_fingerprint = hashlib.sha256(
            encoded_configuration
        ).hexdigest()

        os.makedirs(self.progress_folder, exist_ok=True)
        os.makedirs(self.generations_directory, exist_ok=True)
        self._remove_abandoned_transactions()
        if not os.path.islink(self.current_link):
            if os.path.lexists(self.current_link):
                raise RuntimeError(
                    f"Checkpoint CURRENT path is not a symlink: "
                    f"{self.current_link}"
                )
            self._migrate_or_initialize_legacy_checkpoint()
        self._ensure_compatibility_links()
        self.validate_current_generation()

    @property
    def current_generation(self) -> str:
        if not os.path.islink(self.current_link):
            raise RuntimeError(
                f"Missing checkpoint CURRENT symlink: {self.current_link}"
            )
        relative_target = os.readlink(self.current_link)
        generation = os.path.abspath(
            os.path.join(self.progress_folder, relative_target)
        )
        generations_root = os.path.abspath(self.generations_directory)
        if os.path.commonpath((generation, generations_root)) != generations_root:
            raise RuntimeError(
                f"Checkpoint CURRENT escapes its generation directory: "
                f"{relative_target}"
            )
        if os.path.islink(generation):
            raise RuntimeError(
                f"Checkpoint generation target must not be a symlink: "
                f"{generation}"
            )
        real_generation = os.path.realpath(generation)
        real_generations_root = os.path.realpath(generations_root)
        if (
            os.path.commonpath((real_generation, real_generations_root))
            != real_generations_root
        ):
            raise RuntimeError(
                "Checkpoint CURRENT resolves outside its generation "
                f"directory: {relative_target}"
            )
        if not os.path.isdir(generation):
            raise RuntimeError(
                f"Checkpoint CURRENT target is missing: {generation}"
            )
        return generation

    @property
    def shots_remaining(self) -> int:
        shots_path = os.path.join(
            self.current_generation,
            self.SHOTS_FILENAME,
        )
        with open(shots_path, "r") as reader:
            value = int(reader.read().strip())
        if value < 0 or value > self.num_shots:
            raise RuntimeError(
                f"Invalid checkpoint shots remaining {value}; expected "
                f"0..{self.num_shots}."
            )
        return value

    @property
    def current_data_folder(self) -> str:
        return self.current_generation + os.sep

    def _remove_abandoned_transactions(self) -> None:
        for entry in os.scandir(self.generations_directory):
            if entry.name.startswith(".txn-"):
                if entry.is_dir(follow_symlinks=False):
                    shutil.rmtree(entry.path)
                else:
                    os.unlink(entry.path)

    def _manifest(self, shots_remaining: int, batch_index: int) -> dict[str, Any]:
        return {
            "configuration": self.configuration,
            "configuration_fingerprint": self.configuration_fingerprint,
            "shots_remaining": int(shots_remaining),
            "committed_shots": self.num_shots - int(shots_remaining),
            "last_batch_index": int(batch_index),
            "committed_utc": time.strftime(
                "%Y-%m-%dT%H:%M:%SZ",
                time.gmtime(),
            ),
        }

    def _write_manifest(
        self,
        generation: str,
        *,
        shots_remaining: int,
        batch_index: int,
    ) -> None:
        atomic_write_text(
            os.path.join(generation, self.MANIFEST_FILENAME),
            json.dumps(
                self._manifest(shots_remaining, batch_index),
                sort_keys=True,
                indent=2,
            ) + "\n",
        )

    def _publish_current_generation(self, generation: str) -> None:
        relative_target = os.path.relpath(
            generation,
            self.progress_folder,
        )
        temporary_link = os.path.join(
            self.progress_folder,
            f".checkpoint_current.{uuid.uuid4().hex}.tmp",
        )
        os.symlink(relative_target, temporary_link)
        try:
            os.replace(temporary_link, self.current_link)
            _fsync_directory(self.progress_folder)
        finally:
            if os.path.lexists(temporary_link):
                os.unlink(temporary_link)

    def _migrate_or_initialize_legacy_checkpoint(self) -> None:
        legacy_shots_path = os.path.join(
            self.progress_folder,
            self.SHOTS_FILENAME,
        )
        if os.path.exists(legacy_shots_path):
            with open(legacy_shots_path, "r") as reader:
                shots_remaining = int(reader.read().strip())
        else:
            shots_remaining = self.num_shots
        if shots_remaining < 0 or shots_remaining > self.num_shots:
            raise RuntimeError(
                f"Invalid legacy shots remaining {shots_remaining}."
            )

        committed_shots = self.num_shots - shots_remaining
        if committed_shots != 0:
            raise RuntimeError(
                "A committed legacy checkpoint has no recorded source, "
                "circuit, or runtime-profile provenance. Quarantine it "
                "instead of silently mixing it into this run."
            )
        if (
            committed_shots not in (0, self.num_shots)
            and committed_shots % self.batch_size != 0
        ):
            raise RuntimeError(
                f"Legacy checkpoint contains {committed_shots} committed "
                f"shots, which is not aligned to batch size "
                f"{self.batch_size}."
            )
        legacy_has_counts = any(
            os.path.isdir(os.path.join(self.progress_folder, directory))
            and any(
                filename.endswith(".npy")
                for filename in os.listdir(
                    os.path.join(self.progress_folder, directory)
                )
            )
            for directory in self.COUNT_DIRECTORIES
        )
        if committed_shots == 0 and legacy_has_counts:
            raise RuntimeError(
                "Legacy checkpoint has count arrays but reports zero "
                "committed shots. Quarantine the interrupted arrays before "
                "restarting this probability index."
            )

        transaction = tempfile.mkdtemp(
            dir=self.generations_directory,
            prefix=".txn-initialize-",
        )
        try:
            for directory in self.COUNT_DIRECTORIES:
                source = os.path.join(self.progress_folder, directory)
                destination = os.path.join(transaction, directory)
                if os.path.isdir(source):
                    shutil.copytree(source, destination)
                else:
                    os.makedirs(destination)

            atomic_write_text(
                os.path.join(transaction, self.SHOTS_FILENAME),
                str(shots_remaining),
            )
            self._write_manifest(
                transaction,
                shots_remaining=shots_remaining,
                batch_index=(
                    (committed_shots + self.batch_size - 1)
                    // self.batch_size
                ),
            )
            self.validate_generation(
                transaction,
                expected_committed_shots=committed_shots,
            )
            _fsync_tree(transaction)
            generation = os.path.join(
                self.generations_directory,
                f"gen-{committed_shots:09d}-{uuid.uuid4().hex}",
            )
            os.replace(transaction, generation)
            _fsync_directory(self.generations_directory)
            self._publish_current_generation(generation)
        except BaseException:
            if os.path.isdir(transaction):
                shutil.rmtree(transaction)
            raise

    def _ensure_compatibility_links(self) -> None:
        archive_root = os.path.join(
            self.progress_folder,
            ".legacy_checkpoint_migrated",
        )
        aliases = {
            self.SHOTS_FILENAME: os.path.join(
                self.CURRENT_LINK,
                self.SHOTS_FILENAME,
            ),
            **{
                directory: os.path.join(self.CURRENT_LINK, directory)
                for directory in self.COUNT_DIRECTORIES
            },
        }
        for alias_name, link_target in aliases.items():
            alias_path = os.path.join(self.progress_folder, alias_name)
            if os.path.islink(alias_path):
                if os.readlink(alias_path) == link_target:
                    continue
                os.unlink(alias_path)
            elif os.path.lexists(alias_path):
                os.makedirs(archive_root, exist_ok=True)
                archive_path = os.path.join(archive_root, alias_name)
                if os.path.lexists(archive_path):
                    archive_path += f".{uuid.uuid4().hex}"
                os.replace(alias_path, archive_path)

            temporary_link = os.path.join(
                self.progress_folder,
                f".{alias_name}.{uuid.uuid4().hex}.tmp",
            )
            os.symlink(link_target, temporary_link)
            os.replace(temporary_link, alias_path)
        _fsync_directory(self.progress_folder)

    def prepare_batch(
        self,
        *,
        shots_remaining_before: int,
        batch_index: int,
    ) -> str:
        if self._active_transaction is not None:
            raise RuntimeError("A checkpoint transaction is already active.")
        if shots_remaining_before != self.shots_remaining:
            raise RuntimeError(
                "Checkpoint changed before batch start: expected "
                f"{shots_remaining_before}, found {self.shots_remaining}."
            )
        transaction = tempfile.mkdtemp(
            dir=self.generations_directory,
            prefix=f".txn-batch-{batch_index:04d}-",
        )
        shutil.rmtree(transaction)
        shutil.copytree(self.current_generation, transaction)
        self._active_transaction = transaction
        _checkpoint_test_failpoint("after_prepare_copy")
        return transaction + os.sep

    def commit_batch(
        self,
        transaction_folder: str,
        *,
        shots_remaining_after: int,
        batch_index: int,
    ) -> None:
        transaction = os.path.abspath(transaction_folder)
        if transaction != self._active_transaction:
            raise RuntimeError(
                f"Unexpected checkpoint transaction {transaction_folder}."
            )
        shots_remaining_after = int(shots_remaining_after)
        shots_remaining_before = self.shots_remaining
        expected_remaining_after = max(
            0,
            shots_remaining_before - self.batch_size,
        )
        if shots_remaining_after != expected_remaining_after:
            raise RuntimeError(
                f"Invalid checkpoint transition "
                f"{shots_remaining_before}->{shots_remaining_after}; "
                f"expected {shots_remaining_before}->"
                f"{expected_remaining_after}."
            )
        expected_batch_index = (
            (self.num_shots - shots_remaining_before) // self.batch_size
        ) + 1
        if int(batch_index) != expected_batch_index:
            raise RuntimeError(
                f"Invalid checkpoint batch index {batch_index}; expected "
                f"{expected_batch_index}."
            )
        committed_shots = self.num_shots - shots_remaining_after
        self.validate_generation(
            transaction,
            expected_committed_shots=committed_shots,
        )
        atomic_write_text(
            os.path.join(transaction, self.SHOTS_FILENAME),
            str(shots_remaining_after),
        )
        self._write_manifest(
            transaction,
            shots_remaining=shots_remaining_after,
            batch_index=batch_index,
        )
        _fsync_tree(transaction)
        _checkpoint_test_failpoint("after_staged_metadata")
        generation = os.path.join(
            self.generations_directory,
            f"gen-{committed_shots:09d}-{uuid.uuid4().hex}",
        )
        os.replace(transaction, generation)
        _fsync_directory(self.generations_directory)
        _checkpoint_test_failpoint("after_generation_rename")
        _checkpoint_test_failpoint("before_current_replace")
        self._publish_current_generation(generation)
        _checkpoint_test_failpoint("after_current_replace")
        self._active_transaction = None
        self.validate_current_generation()

    def validate_current_generation(self) -> None:
        generation = self.current_generation
        manifest_path = os.path.join(generation, self.MANIFEST_FILENAME)
        with open(manifest_path, "r") as manifest_file:
            manifest = json.load(manifest_file)
        manifest_configuration = manifest.get("configuration")
        if not isinstance(manifest_configuration, dict):
            raise RuntimeError(
                "Checkpoint manifest is missing its configuration."
            )
        encoded_manifest_configuration = json.dumps(
            manifest_configuration,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        recorded_configuration_fingerprint = manifest.get(
            "configuration_fingerprint"
        )
        recomputed_configuration_fingerprint = hashlib.sha256(
            encoded_manifest_configuration
        ).hexdigest()
        if (
            recorded_configuration_fingerprint
            != recomputed_configuration_fingerprint
        ):
            raise RuntimeError(
                "Checkpoint manifest configuration was modified or "
                "corrupted."
            )
        if (
            recorded_configuration_fingerprint
            != self.configuration_fingerprint
        ):
            raise RuntimeError(
                "Checkpoint configuration/source fingerprint changed. "
                "Refusing to mix simulation data from different runtimes."
            )
        shots_remaining = self.shots_remaining
        if manifest.get("shots_remaining") != shots_remaining:
            raise RuntimeError(
                "Checkpoint manifest and shots_remaining.txt disagree."
            )
        if manifest.get("committed_shots") != self.num_shots - shots_remaining:
            raise RuntimeError(
                "Checkpoint manifest committed-shot total is inconsistent."
            )
        committed_shots = self.num_shots - shots_remaining
        expected_last_batch_index = (
            (committed_shots + self.batch_size - 1) // self.batch_size
        )
        if manifest.get("last_batch_index") != expected_last_batch_index:
            raise RuntimeError(
                "Checkpoint manifest batch index is inconsistent with its "
                "committed-shot total."
            )
        self.validate_generation(
            generation,
            expected_committed_shots=self.num_shots - shots_remaining,
        )

    def validate_generation(
        self,
        generation: str,
        *,
        expected_committed_shots: int,
    ) -> None:
        expected_shapes = (
            (len(self.depths), self.num_cliff_seq, self.dimension),
            (len(self.depths), self.num_cliff_seq),
        )
        policies = (
            ("const_check_data", self.const_checks),
            ("unif_check_data", self.unif_checks),
        )
        for directory, checks in policies:
            directory_path = os.path.join(generation, directory)
            if not os.path.isdir(directory_path):
                raise RuntimeError(
                    f"Missing checkpoint count directory {directory_path}."
                )
            expected_files = {
                filename
                for check in checks
                for filename in (f"{check}.npy", f"{check}_rejected.npy")
            }
            actual_files = {
                entry.name
                for entry in os.scandir(directory_path)
                if entry.is_file(follow_symlinks=False)
            }
            if expected_committed_shots == 0 and not actual_files:
                continue
            if actual_files != expected_files:
                missing = sorted(expected_files - actual_files)
                extra = sorted(actual_files - expected_files)
                raise RuntimeError(
                    f"Checkpoint policy files differ in {directory}: "
                    f"missing={missing}, extra={extra}."
                )
            for check in checks:
                counts = np.load(
                    os.path.join(directory_path, f"{check}.npy"),
                    allow_pickle=False,
                )
                rejected = np.load(
                    os.path.join(
                        directory_path,
                        f"{check}_rejected.npy",
                    ),
                    allow_pickle=False,
                )
                if counts.shape != expected_shapes[0]:
                    raise RuntimeError(
                        f"Wrong checkpoint count shape for {directory}/"
                        f"{check}: {counts.shape}, expected "
                        f"{expected_shapes[0]}."
                    )
                if rejected.shape != expected_shapes[1]:
                    raise RuntimeError(
                        f"Wrong checkpoint rejection shape for {directory}/"
                        f"{check}: {rejected.shape}, expected "
                        f"{expected_shapes[1]}."
                    )
                if not np.issubdtype(counts.dtype, np.integer) \
                        or not np.issubdtype(rejected.dtype, np.integer):
                    raise RuntimeError(
                        f"Checkpoint arrays for {directory}/{check} must be "
                        "integer-valued."
                    )
                if np.any(counts < 0) or np.any(rejected < 0):
                    raise RuntimeError(
                        f"Checkpoint arrays for {directory}/{check} contain "
                        "negative values."
                    )
                totals = counts.sum(axis=2, dtype=np.int64) + rejected
                if np.any(totals != expected_committed_shots):
                    observed = np.unique(totals)
                    raise RuntimeError(
                        f"Checkpoint totals for {directory}/{check} are "
                        f"{observed.tolist()}, expected exactly "
                        f"{expected_committed_shots} per cell."
                    )


def open_lrb_checkpoint_store(
    *,
    partial_progress_folder_path: str,
    num_shots: int,
    batch_size: int,
    num_cliff_seq: int,
    depths: Sequence[int],
    stab_checks_const: Sequence[int],
    stab_checks_unif: Sequence[int],
    logical_dimension: int,
    error_prob_ind: int,
    error_prob: float,
    backend: str | None,
    runtime_profile: str,
    filter_trivial_shots: bool,
    lrb_experiment_folder_path: str,
    lrb_const0_experiment_folder_path: str | None,
    rb_experiment_folder_path: str,
) -> tuple[
    LRBBatchCheckpointStore,
    str,
    tuple[tuple[str, str], ...],
]:
    """Open and deeply validate the checkpoint and its exact circuit inputs."""
    resolved_backend = _resolve_simulation_backend(backend)
    const0_requested = 0 in stab_checks_const
    need_main_lrb = bool(
        [check for check in stab_checks_const if check != 0]
        or stab_checks_unif
    )
    circuit_folders: list[tuple[str, str]] = []
    if need_main_lrb:
        circuit_folders.append(("LRB", lrb_experiment_folder_path))
    if const0_requested:
        if lrb_const0_experiment_folder_path is None:
            raise RuntimeError(
                "const=0 checkpoint validation requires LRB_const0 circuits."
            )
        circuit_folders.append(
            ("LRB_const0", lrb_const0_experiment_folder_path)
        )
    circuit_folders.append(("RB", rb_experiment_folder_path))
    immutable_circuit_folders = tuple(circuit_folders)
    input_fingerprint = _circuit_input_fingerprint(
        immutable_circuit_folders,
        num_cliff_seq=num_cliff_seq,
        probability_index=error_prob_ind,
        num_depths=len(depths),
    )
    checkpoint_store = LRBBatchCheckpointStore(
        partial_progress_folder_path,
        num_shots=num_shots,
        batch_size=batch_size,
        num_cliff_seq=num_cliff_seq,
        depths=depths,
        const_checks=stab_checks_const,
        unif_checks=stab_checks_unif,
        dimension=logical_dimension,
        probability_index=error_prob_ind,
        probability=error_prob,
        backend=resolved_backend,
        input_fingerprint=input_fingerprint,
        runtime_profile=runtime_profile,
        filter_trivial_shots=filter_trivial_shots,
    )
    return (
        checkpoint_store,
        input_fingerprint,
        immutable_circuit_folders,
    )


def _circuit_timing_context(circuit) -> dict[str, Any]:
    operations = tuple(getattr(circuit, "operations", ()))
    circuit_qudits = getattr(circuit, "num_qudits", None)
    if callable(circuit_qudits):
        circuit_qudits = circuit_qudits()
    if circuit_qudits is None:
        max_qudit = -1
        for operation in operations:
            for attr in ("qudit_index", "target_index"):
                index = getattr(operation, attr, None)
                if index is not None:
                    max_qudit = max(max_qudit, int(index))
        circuit_qudits = max_qudit + 1 if max_qudit >= 0 else ""

    return {
        "circuit_dimension": getattr(circuit, "dimension", ""),
        "circuit_qudits": circuit_qudits,
        "circuit_operations": len(operations),
    }


def _require_sdim_runtime() -> None:
    """Raise a clear error when SDIM simulation runtime is unavailable."""
    if _SDIM_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "SDIM runtime is unavailable in this environment. "
        "Simulation execution methods require a compatible SDIM install."
    ) from _SDIM_IMPORT_ERROR


class NoiseModelUtils:
    """
    Utilities for normalizing and validating noise-model parameters.

    Attributes:
        None: This class is stateless and stores no persistent attributes.

    Methods:
        two_qudit_depol_prob_dist(dimension, prob): Build a full two-qudit
            depolarizing probability distribution.
        ensure_noise_params(circuit, default_prob): Ensure N2 operations carry
            a valid ``prob_dist`` payload.
    """

    @staticmethod
    def two_qudit_depol_prob_dist(dimension: int, prob: float):
        """
        Two qudit depol prob dist.
        
        Args:
            dimension (Any): Input argument.
            prob (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        n = dimension**4
        if n <= 1:
            return [1.0]
        if prob < 0.0 or prob > 1.0:
            raise ValueError(f"Invalid depolarizing prob {prob}")
        tail = prob / (n - 1)
        dist = [tail] * n
        dist[0] = 1.0 - prob
        return dist

    @staticmethod
    def ensure_noise_params(circuit, default_prob: float = None):
        """
        Ensure noise params.
        
        Args:
            circuit (Any): Input argument.
            default_prob (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        for op in circuit.operations:
            if op.gate_name != 'N2' and getattr(op, 'gate_id', None) != 18:
                continue
            if op.params is None:
                op.params = {}

            prob_dist = op.params.get('prob_dist', None)
            if isinstance(prob_dist, str):
                try:
                    parsed = ast.literal_eval(prob_dist)
                    op.params['prob_dist'] = parsed
                    prob_dist = parsed
                except Exception:
                    prob_dist = None

            if isinstance(
                    prob_dist,
                (list, tuple)) and len(prob_dist) == circuit.dimension**4:
                continue

            prob = op.params.get('prob', default_prob)
            if prob is None:
                raise ValueError(
                    "N2 gate missing 'prob' and 'prob_dist' params; "
                    "cannot expand."
                )
            op.params['prob_dist'] = NoiseModelUtils.two_qudit_depol_prob_dist(
                circuit.dimension, float(prob))


@dataclass(frozen=True)
class LRBUnpackSpec:
    """
    Specification describing logical/stabilizer unpack behavior.

    Attributes:
        stabilizer_wires (tuple[int, ...]): Wires measured for stabilizer
            checks.
        logical_measurement_wires (tuple[int, ...]): Wires used for logical
            readout.
        logical_outcome_fn (Callable[[list[int], int], int]): Function mapping
            logical measurements and depth to an integer logical outcome.
        terminal_x_measurement_wires (tuple[int, ...]): Data wires measured
            by the direct terminal X readout for ``const=0``.
        x_stabilizer_check_fn (Callable[[list[int]], bool]): Function that
            evaluates the X-stabilizer filter from direct X measurements.
        logical_observable_terms (tuple[tuple[int, int], ...]): Terminal
            logical observable as ``(wire, coefficient)`` terms. Detector
            unpacking uses the matching generated SDIM logical observable and
            falls back to this term list for the reference shot.
        terminal_x_stabilizer_terms (tuple[tuple[tuple[int, int], ...], ...]):
            Direct terminal X-stabilizer expressions used to reconstruct the
            reference shot for detector-backed ``const=0`` unpacking.
        stabilizer_pass_fn (Callable[[list[int]], bool] | None): Optional
            function that maps one stabilizer-check layer's ancilla readouts to
            a full pass/fail decision. ``None`` keeps the legacy behavior where
            every listed ancilla result must be zero modulo the code dimension.
        check_stride (int): Round-to-round stride in measurement records.
        check_round_start (int): First check round index to inspect.
        check_rounds_offset (int): Offset applied to depth for round count.
        logical_measurement_round (int): Round index used for logical readout.
        dimension (int): Local qudit dimension used for detector/logical
            modulo arithmetic.

    Methods:
        This dataclass is declarative and defines no custom methods.
    """

    stabilizer_wires: tuple[int, ...]
    logical_measurement_wires: tuple[int, ...]
    logical_outcome_fn: Callable[[list[int], int], int]
    terminal_x_measurement_wires: tuple[int, ...]
    x_stabilizer_check_fn: Callable[[list[int]], bool]
    logical_observable_terms: tuple[tuple[int, int], ...]
    terminal_x_stabilizer_terms: tuple[tuple[tuple[int, int], ...], ...]
    stabilizer_pass_fn: Callable[[list[int]], bool] | None = None
    check_stride: int = 2
    check_round_start: int = 0
    check_rounds_offset: int = 1
    logical_measurement_round: int = 0
    dimension: int = 3


@dataclass
class LRBSimulationEngine:
    """
    Object-oriented facade for RB/LRB simulation workflows.

    Attributes:
        normal_rb_shots (int): Default shot count used by physical RB runs.

    Methods:
        ensure_noise_parameters(circuit, default_prob): Normalize N2 noise
            parameters in one circuit.
        run_lrb(...): Execute logical RB simulations for prepared circuits.
        run_rb(...): Execute physical RB simulations for prepared circuits.
        evaluate_uniform_postselection(...): Build uniform-interval acceptance
            table.
        evaluate_constant_postselection(...): Build constant-count acceptance
            table.
        apply_postselection(...): Apply acceptance decisions to measurement
            records.
        run_round(...): Execute one resumable probability-index experiment
            round.
    """

    normal_rb_shots: int = NORMAL_RB_SHOTS

    def ensure_noise_parameters(self,
                                circuit: Circuit,
                                default_prob: float | None = None) -> None:
        """
        Ensure noise parameters.
        
        Args:
            circuit (Any): Input argument.
            default_prob (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        NoiseModelUtils.ensure_noise_params(circuit, default_prob=default_prob)

    def run_lrb(self,
                experiments,
                depths: list[int],
                shots: int,
                unpack_func: Callable | None = None,
                partial_progress_folder: str = "./prog",
                simulation_backend: str | None = None):
        """
        Run lrb.
        
        Args:
            experiments (Any): Input argument.
            depths (Any): Input argument.
            shots (Any): Input argument.
            unpack_func (Any): Input argument.
            partial_progress_folder (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        if unpack_func is None:
            raise ValueError("run_lrb requires an explicit unpack_func.")
        return LRBSimulationPipeline.LRB(
            experiments=experiments,
            depths=depths,
            shots=shots,
            unpack_func=unpack_func,
            partial_progress_folder=partial_progress_folder,
            simulation_backend=simulation_backend,
        )

    def run_rb(self,
               experiments,
               depths: list[int],
               shots: int | None = None,
               simulation_backend: str | None = None) -> np.ndarray:
        """
        Run rb.
        
        Args:
            experiments (Any): Input argument.
            depths (Any): Input argument.
            shots (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        actual_shots = self.normal_rb_shots if shots is None else shots
        return LRBSimulationPipeline.RB(experiments=experiments,
                                        depths=depths,
                                        shots=actual_shots,
                                        simulation_backend=simulation_backend)

    def evaluate_uniform_postselection(
        self,
        stabilizer_check_record: np.ndarray,
        depths: list[int],
        interval: int,
    ):
        """
        Evaluate uniform postselection.
        
        Args:
            stabilizer_check_record (Any): Input argument.
            depths (Any): Input argument.
            interval (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        return LRBSimulationPipeline.postselect_uniform_interval(
            stabilizer_check_record, depths, interval)

    def evaluate_constant_postselection(self,
                                        stabilizer_check_record: np.ndarray,
                                        depths: list[int],
                                        num_checks: int):
        """
        Evaluate constant postselection.
        
        Args:
            stabilizer_check_record (Any): Input argument.
            depths (Any): Input argument.
            num_checks (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        return LRBSimulationPipeline.postselect_constant_number(
            stabilizer_check_record, depths, num_checks)

    def apply_postselection(self, measurement_record: np.ndarray,
                            decision_table: np.ndarray):
        """
        Apply postselection.
        
        Args:
            measurement_record (Any): Input argument.
            decision_table (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        return LRBSimulationPipeline.postselect_record(measurement_record,
                                                       decision_table)

    def run_round(
        self,
        stab_checks_const,
        stab_checks_unif,
        batch_size: int,
        error_prob: float,
        error_prob_ind: int,
        num_cliff_seq: int,
        depths: list[int],
        num_shots: int,
        filter_trivial_shots: bool,
        lrb_experiment_folder_path: str,
        rb_experiment_folder_path: str,
        lrb_results_folder_path: str,
        rb_results_folder_path: str,
        partial_progress_folder_path: str,
        unpack_func: Callable | None = None,
        const0_unpack_func: Callable | None = None,
        logical_dimension: int = 3,
        lrb_const0_experiment_folder_path: str | None = None,
        simulation_backend: str | None = None,
        runtime_profile: str = "unspecified",
    ) -> int:
        """
        Run round.
        
        Args:
            stab_checks_const (Any): Input argument.
            stab_checks_unif (Any): Input argument.
            batch_size (Any): Input argument.
            error_prob (Any): Input argument.
            error_prob_ind (Any): Input argument.
            num_cliff_seq (Any): Input argument.
            depths (Any): Input argument.
            num_shots (Any): Input argument.
            filter_trivial_shots (Any): Input argument.
            lrb_experiment_folder_path (Any): Input argument.
            rb_experiment_folder_path (Any): Input argument.
            lrb_results_folder_path (Any): Input argument.
            rb_results_folder_path (Any): Input argument.
            partial_progress_folder_path (Any): Input argument.
            unpack_func (Any): Input argument.
            logical_dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        if unpack_func is None:
            raise ValueError("run_round requires an explicit unpack_func.")
        return LRBSimulationPipeline.run_LRB_round(
            stab_checks_const=stab_checks_const,
            stab_checks_unif=stab_checks_unif,
            BATCH_SIZE=batch_size,
            error_prob=error_prob,
            error_prob_ind=error_prob_ind,
            num_cliff_seq=num_cliff_seq,
            depths=depths,
            num_shots=num_shots,
            filter_trivial_shots=filter_trivial_shots,
            LRB_experiment_folder_path=lrb_experiment_folder_path,
            RB_experiment_folder_path=rb_experiment_folder_path,
            LRB_results_folder_path=lrb_results_folder_path,
            RB_results_folder_path=rb_results_folder_path,
            partial_progress_folder_path=partial_progress_folder_path,
            unpack_func=unpack_func,
            const0_unpack_func=const0_unpack_func,
            logical_dimension=logical_dimension,
            LRB_const0_experiment_folder_path=(
                lrb_const0_experiment_folder_path),
            simulation_backend=simulation_backend,
            runtime_profile=runtime_profile,
        )


class LRBSimulationPipeline:
    """
    Static helpers for simulation, postselection, and statistics.

    Attributes:
        None: This class is stateless and stores no persistent attributes.

    Methods:
        unpack_measurement_results_from_spec(...): Generic measurement
            unpacker.
        LRB/RB(...): Execute logical or physical RB simulations.
        postselect_* methods: Build and apply stabilizer-based acceptance
            tables.
        extract_* and *_stats methods: Convert raw records into summary
            statistics.
        run_LRB_round(...): Execute one full resumable LRB/RB round.
    """
    @staticmethod
    def default_stabilizer_pass_check(
            stabilizer_measurements: list[int]) -> bool:
        """
        Evaluate the legacy one-readout-per-stabilizer pass condition.

        The original folded and QGRM profiles store one qutrit-valued ancilla
        result for each stabilizer generator in a check layer. A check layer
        passes exactly when every listed stabilizer syndrome is zero in
        ``Z_3``. Older code used a direct ``value == 0`` comparison; this helper
        keeps the same behavior for valid SDIM qutrit measurement values while
        making the modulo-three syndrome convention explicit.

        Args:
            stabilizer_measurements (list[int]): Stabilizer ancilla readouts
                from one check layer.

        Returns:
            bool: ``True`` when every readout is zero modulo three.

        Raises:
            ValueError: Not raised directly by this helper.
        """
        return all((value % 3) == 0 for value in stabilizer_measurements)

    @staticmethod
    def detector_values_from_reference(
        reference_value: int,
        detector_shifts: Sequence[int] | np.ndarray,
        shots: int,
        dimension: int,
    ) -> np.ndarray:
        """
        Reconstruct absolute qutrit values from an SDIM detector-shift vector.

        In SDIM frame simulation, detector and logical-observable arrays store
        the ``shots - 1`` Pauli-frame shifts relative to the single reference
        tableau shot. The older LRB post-processing code consumes absolute
        qutrit-valued outcomes for all requested shots. This helper bridges
        those representations by placing the reference value in slot zero and
        adding SDIM's shift vector modulo the local dimension for the remaining
        shots.

        Args:
            reference_value (int): Absolute qutrit value from the reference
                shot.
            detector_shifts (Sequence[int] | np.ndarray): SDIM detector or
                logical-observable shift data for shots ``1..shots-1``.
            shots (int): Total requested shot count.
            dimension (int): Local qudit dimension.

        Returns:
            np.ndarray: One-dimensional integer array of length ``shots``.

        Raises:
            ValueError: If the detector shift length is incompatible with the
                requested shot count.
        """
        shift_array = np.asarray(detector_shifts, dtype=np.int64)
        expected_extra_shots = max(int(shots) - 1, 0)
        if shift_array.size != expected_extra_shots:
            raise ValueError(
                "Detector shift vector length does not match the requested "
                f"shot count: got {shift_array.size}, expected "
                f"{expected_extra_shots}."
            )

        values = np.empty(shots, dtype=np.int64)
        values[0] = int(reference_value) % int(dimension)
        if shots > 1:
            values[1:] = (
                int(reference_value) + shift_array
            ) % int(dimension)
        return values

    @staticmethod
    def reference_linear_value_from_results(
        results,
        wire_terms: Sequence[tuple[int, int]],
        measurement_round: int,
        dimension: int,
    ) -> int:
        """
        Evaluate a linear qutrit expression on the reference raw result shot.

        Detector arrays provide only frame shifts for the vectorized shots.
        The reference shot is still taken from SDIM's ordinary measurement
        records. This helper evaluates the same linear expression used by a
        generated detector or logical observable on shot zero, so the detector
        shift vector can be converted back into absolute qutrit values.

        Args:
            results (Any): Raw SDIM measurement-result tensor.
            wire_terms (Sequence[tuple[int, int]]): ``(wire, coefficient)``
                terms defining the linear expression.
            measurement_round (int): Per-wire raw measurement-result round.
            dimension (int): Local qudit dimension.

        Returns:
            int: Expression value on reference shot zero, reduced modulo
            ``dimension``.

        Raises:
            IndexError: If a required wire or measurement round is missing.
        """
        value = 0
        for wire, coefficient in wire_terms:
            value += (
                int(coefficient)
                * int(results[int(wire)][measurement_round][0]
                      .measurement_value)
            )
        return value % int(dimension)

    @staticmethod
    def try_unpack_detector_results_from_spec(
        simulation_output,
        depth: int,
        shots: int,
        spec: LRBUnpackSpec,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Try to unpack ordinary LRB data from SDIM detector/logical arrays.

        This is the detector-backed counterpart to the raw measurement unpack
        path. It returns ``None`` when the expected labels are absent, which
        lets older generated circuits and custom tests continue through the
        legacy raw-measurement logic.

        Args:
            simulation_output (Any): Direct return from
                ``Program(...).simulate(...)``.
            depth (int): Benchmark depth for the circuit.
            shots (int): Number of simulated shots.
            spec (LRBUnpackSpec): Code-specific unpack and detector layout.

        Returns:
            tuple[np.ndarray, np.ndarray] | None: Stabilizer decisions with
            shape ``(shots, depth + check_rounds_offset)`` and logical values
            with shape ``(shots,)`` when detector data is usable; otherwise
            ``None``.

        Raises:
            ValueError: If present detector arrays have incompatible lengths.
        """
        compact_results = (
            LRBSimulationPipeline.
            compact_detector_results_from_sdim_output(simulation_output)
        )
        check_round_limit = depth + spec.check_rounds_offset
        required_detector_labels = [
            lrb_stabilizer_detector_label(check_round, wire)
            for check_round in range(
                spec.check_round_start, check_round_limit)
            for wire in spec.stabilizer_wires
        ]
        if any(
            label not in compact_results["detectors"]
            for label in required_detector_labels
        ):
            return None
        if LRB_LOGICAL_OBSERVABLE_LABEL not in compact_results["logicals"]:
            return None

        results = LRBSimulationPipeline.measurement_results_from_sdim_output(
            simulation_output)
        stabilizer_check_record = np.zeros(
            (shots, check_round_limit - spec.check_round_start),
            dtype=np.int64,
        )

        for output_round, check_round in enumerate(
                range(spec.check_round_start, check_round_limit)):
            reconstructed_values: list[np.ndarray] = []
            check_offset = check_round * spec.check_stride
            for wire in spec.stabilizer_wires:
                reference_value = int(
                    results[wire][check_offset][0].measurement_value)
                label = lrb_stabilizer_detector_label(check_round, wire)
                reconstructed_values.append(
                    LRBSimulationPipeline.detector_values_from_reference(
                        reference_value=reference_value,
                        detector_shifts=compact_results["detectors"][label],
                        shots=shots,
                        dimension=spec.dimension,
                    )
                )

            values_by_wire = np.stack(reconstructed_values, axis=1)
            stabilizer_check_record[:, output_round] = (
                np.all((values_by_wire % spec.dimension) == 0, axis=1)
                .astype(np.int64)
            )

        reference_logical_value = (
            LRBSimulationPipeline.reference_linear_value_from_results(
                results=results,
                wire_terms=spec.logical_observable_terms,
                measurement_round=spec.logical_measurement_round,
                dimension=spec.dimension,
            )
        )
        logical_values = LRBSimulationPipeline.detector_values_from_reference(
            reference_value=reference_logical_value,
            detector_shifts=(
                compact_results["logicals"][LRB_LOGICAL_OBSERVABLE_LABEL]),
            shots=shots,
            dimension=spec.dimension,
        )
        return stabilizer_check_record, logical_values

    @staticmethod
    def try_unpack_const0_detector_results_from_spec(
        simulation_output,
        depth: int,
        shots: int,
        spec: LRBUnpackSpec,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Try to unpack ``const=0`` direct-X data from detector/logical arrays.

        ``const=0`` circuits have no intermediate ancilla checks. The only
        postselection decision is placed in slot ``depth`` and is computed from
        direct X-basis data measurements at the end of the circuit. This method
        reconstructs those terminal X-stabilizer values from compact detector
        arrays and preserves the legacy table shape.

        Args:
            simulation_output (Any): Direct return from
                ``Program(...).simulate(...)``.
            depth (int): Benchmark depth for the current circuit.
            shots (int): Number of simulated shots.
            spec (LRBUnpackSpec): Code-specific direct-X readout layout.

        Returns:
            tuple[np.ndarray, np.ndarray] | None: Stabilizer decisions with
            shape ``(shots, depth + 1)`` and logical values with shape
            ``(shots,)`` when detector data is usable; otherwise ``None``.

        Raises:
            ValueError: If present detector arrays have incompatible lengths.
        """
        compact_results = (
            LRBSimulationPipeline.
            compact_detector_results_from_sdim_output(simulation_output)
        )
        required_detector_labels = [
            lrb_const0_stabilizer_detector_label(stabilizer_index)
            for stabilizer_index in range(
                len(spec.terminal_x_stabilizer_terms))
        ]
        if any(
            label not in compact_results["detectors"]
            for label in required_detector_labels
        ):
            return None
        if (LRB_CONST0_LOGICAL_OBSERVABLE_LABEL
                not in compact_results["logicals"]):
            return None

        results = LRBSimulationPipeline.measurement_results_from_sdim_output(
            simulation_output)
        reconstructed_stabilizers: list[np.ndarray] = []
        for stabilizer_index, stabilizer_terms in enumerate(
                spec.terminal_x_stabilizer_terms):
            reference_value = (
                LRBSimulationPipeline.reference_linear_value_from_results(
                    results=results,
                    wire_terms=stabilizer_terms,
                    measurement_round=0,
                    dimension=spec.dimension,
                )
            )
            label = lrb_const0_stabilizer_detector_label(stabilizer_index)
            reconstructed_stabilizers.append(
                LRBSimulationPipeline.detector_values_from_reference(
                    reference_value=reference_value,
                    detector_shifts=compact_results["detectors"][label],
                    shots=shots,
                    dimension=spec.dimension,
                )
            )

        stabilizer_check_record = np.zeros(
            (shots, depth + 1),
            dtype=np.int64,
        )
        values_by_stabilizer = np.stack(
            reconstructed_stabilizers, axis=1)
        stabilizer_check_record[:, depth] = (
            np.all(
                (values_by_stabilizer % spec.dimension) == 0,
                axis=1,
            ).astype(np.int64)
        )

        reference_logical_value = (
            LRBSimulationPipeline.reference_linear_value_from_results(
                results=results,
                wire_terms=spec.logical_observable_terms,
                measurement_round=0,
                dimension=spec.dimension,
            )
        )
        logical_values = LRBSimulationPipeline.detector_values_from_reference(
            reference_value=reference_logical_value,
            detector_shifts=(
                compact_results["logicals"][
                    LRB_CONST0_LOGICAL_OBSERVABLE_LABEL]),
            shots=shots,
            dimension=spec.dimension,
        )
        return stabilizer_check_record, logical_values

    @staticmethod
    def unpack_measurement_results_from_spec(
        results,
        depth: int,
        shots: int,
        spec: LRBUnpackSpec,
    ) -> tuple[list[list[int]] | np.ndarray, list[int] | np.ndarray]:
        """
        Unpack stabilizer checks and logical outcomes from simulator results.

        Args:
            results (Any): Raw simulator measurement tensor.
            depth (int): Benchmark depth for the current circuit.
            shots (int): Number of simulated shots.
            spec (LRBUnpackSpec): Wire and indexing specification used to
                unpack results.

        Returns:
            tuple[list[list[int]], list[int]]: Stabilizer-pass decisions per
            shot and logical outcomes per shot.

        Raises:
            IndexError: If unpack indices are incompatible with result shape.
            ValueError: If logical outcome mapping cannot produce integers.
        """
        detector_unpack = (
            LRBSimulationPipeline.try_unpack_detector_results_from_spec(
                results, depth, shots, spec
            )
        )
        if detector_unpack is not None:
            return detector_unpack

        results = LRBSimulationPipeline.measurement_results_from_sdim_output(
            results)
        accept_decisions: list[list[int]] = []
        measurement_values: list[int] = []
        check_round_limit = depth + spec.check_rounds_offset
        stabilizer_pass_fn = (
            spec.stabilizer_pass_fn
            if spec.stabilizer_pass_fn is not None
            else LRBSimulationPipeline.default_stabilizer_pass_check
        )
    
        for shot_idx in range(shots):
            shot_checks: list[int] = []
    
            for check_round in range(
                spec.check_round_start, check_round_limit
            ):
                check_offset = check_round * spec.check_stride
                stab_values = [
                    results[wire][check_offset][shot_idx].measurement_value
                    for wire in spec.stabilizer_wires
                ]
                shot_checks.append(int(stabilizer_pass_fn(stab_values)))
    
            logical_measurements = [
                results[wire][spec.logical_measurement_round][shot_idx]
                .measurement_value for wire in spec.logical_measurement_wires
            ]
            measurement_values.append(
                int(spec.logical_outcome_fn(logical_measurements, depth)))
            accept_decisions.append(shot_checks)
    
        return accept_decisions, measurement_values

    @staticmethod
    def unpack_const0_direct_x_results_from_spec(
        results,
        depth: int,
        shots: int,
        spec: LRBUnpackSpec,
    ) -> tuple[list[list[int]] | np.ndarray, list[int] | np.ndarray]:
        """
        Unpack direct terminal X-data measurements for ``const=0`` runs.

        Args:
            results (Any): Raw simulator measurement tensor.
            depth (int): Benchmark depth for the current circuit.
            shots (int): Number of simulated shots.
            spec (LRBUnpackSpec): Wire and decoding specification.

        Returns:
            tuple[list[list[int]], list[int]]: Stabilizer-pass vectors with
            only the terminal slot populated, and logical outcomes per shot.
        """
        detector_unpack = (
            LRBSimulationPipeline.
            try_unpack_const0_detector_results_from_spec(
                results, depth, shots, spec
            )
        )
        if detector_unpack is not None:
            return detector_unpack

        results = LRBSimulationPipeline.measurement_results_from_sdim_output(
            results)
        accept_decisions: list[list[int]] = []
        measurement_values: list[int] = []

        for shot_idx in range(shots):
            x_measurements = [
                results[wire][0][shot_idx].measurement_value
                for wire in spec.terminal_x_measurement_wires
            ]
            measurement_by_wire = dict(
                zip(spec.terminal_x_measurement_wires, x_measurements)
            )
            logical_measurements = [
                measurement_by_wire[wire]
                for wire in spec.logical_measurement_wires
            ]

            shot_checks = [0] * (depth + 1)
            shot_checks[depth] = int(
                spec.x_stabilizer_check_fn(x_measurements))
            accept_decisions.append(shot_checks)
            measurement_values.append(
                int(spec.logical_outcome_fn(logical_measurements, depth)))

        return accept_decisions, measurement_values

    @staticmethod
    def measurement_results_from_sdim_output(simulation_output):
        """
        Return the measurement-record portion of an SDIM simulation result.

        SDIM 1.3.4 returns ``(measurements, detector_results)`` for
        multi-shot frame simulations, even when the circuit has no detectors.
        Older SDIM versions returned the measurement records directly. The
        LRB/RB statistics path consumes only measurements for now.
        """
        if isinstance(simulation_output, tuple):
            return simulation_output[0]
        return simulation_output

    @staticmethod
    def compact_detector_results_from_sdim_output(
            simulation_output) -> dict[str, dict[str, np.ndarray]]:
        """
        Return SDIM detector/logical event arrays in a label-indexed layout.

        SDIM frame simulation can return a two-part object,
        ``(measurements, detector_results)``, when a circuit contains
        ``DETECTOR`` or ``LOGICAL_OBSERVABLE`` operations. The second entry is
        already vectorized internally, but its native shape is a pair of lists:
        one list for detector channels and one list for logical-observable
        channels. Each item in those lists stores a human label and a NumPy
        array of qutrit-valued event data. This helper converts that structure
        into dictionaries keyed by label so experiments can ask directly for
        ``compact["detectors"]["some_label"]`` or
        ``compact["logicals"]["logical_x"]`` without scanning the native event
        lists on every post-processing pass.

        The helper deliberately preserves SDIM's raw frame-simulation shot
        convention. In SDIM 1.3.4, detector/logical arrays observed in
        multi-shot frame simulation contain the extra frame shots and therefore
        have length ``shots - 1``; the reference tableau shot remains in the
        ordinary measurement records. Keeping that convention visible is safer
        than padding or silently inventing a reference-shot detector value at
        this layer.

        Args:
            simulation_output (Any): Direct return value from
                ``Program(circuit).simulate(...)``.

        Returns:
            dict[str, dict[str, np.ndarray]]: A compact result with two top
            level keys, ``"detectors"`` and ``"logicals"``. Each maps stable
            event labels to one-dimensional NumPy arrays. When SDIM returns no
            detector payload, both dictionaries are empty.

        Raises:
            TypeError: If SDIM returns a detector payload in an unexpected
                non-dictionary shape.
        """
        compact_results: dict[str, dict[str, np.ndarray]] = {
            "detectors": {},
            "logicals": {},
        }
        if not isinstance(simulation_output, tuple):
            return compact_results

        detector_payload = simulation_output[1]
        if detector_payload is None:
            return compact_results
        if not isinstance(detector_payload, dict):
            raise TypeError(
                "Expected SDIM detector payload to be a dictionary with "
                "'detectors' and 'logicals' entries."
            )

        for result_kind in ("detectors", "logicals"):
            for event_index, event in enumerate(
                    detector_payload.get(result_kind, [])):
                label = str(event.get("label", f"{result_kind}_{event_index}"))
                if label in compact_results[result_kind]:
                    label = f"{label}#{event_index}"
                compact_results[result_kind][label] = np.asarray(
                    event.get("data", []),
                    dtype=np.int64,
                )

        return compact_results

    @staticmethod
    def simulate_circuit(
        circuit,
        shots: int,
        simulation_backend: str | None = None,
        return_metrics: bool = False,
    ):
        """
        Simulate one SDIM circuit with the selected backend.

        The ``dem`` backend returns the same high-level tuple shape as SDIM's
        Pauli-frame path: reference measurements plus detector/logical frame
        shifts for the remaining shots.
        """
        backend = _resolve_simulation_backend(simulation_backend)
        start_time = time.perf_counter()
        if backend == "sdim":
            simulation_output = Program(circuit).simulate(shots=shots)
            if not return_metrics:
                return simulation_output
            return simulation_output, {
                "backend": backend,
                "simulator_seconds": time.perf_counter() - start_time,
            }

        from .dem_simulation import simulate_circuit_with_dem

        if not return_metrics:
            return simulate_circuit_with_dem(
                circuit,
                shots,
                response_batch_size=_dem_response_batch_size(),
            )

        simulation_output, metrics = simulate_circuit_with_dem(
            circuit,
            shots,
            response_batch_size=_dem_response_batch_size(),
            return_metrics=True,
        )
        metrics["backend"] = backend
        metrics["simulator_seconds"] = time.perf_counter() - start_time
        return simulation_output, metrics

    @staticmethod
    def LRB(
            experiments,
            depths: list[int],
            shots: int,
            unpack_func: Callable,
            partial_progress_folder='./prog',
            simulation_backend: str | None = None,
            timing_phase: str = "LRB",
            probability_index: int | None = None,
            probability: float | None = None,
            batch_index: int | None = None):
        # 2D circuit table: first index is Clifford index, second is
        # experiment index, and entry i uses depth depths[i].
    
        """
        LRB.
        
        Args:
            experiments (Any): Input argument.
            depths (Any): Input argument.
            shots (Any): Input argument.
            unpack_func (Any): Input argument.
            partial_progress_folder (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        _require_sdim_runtime()
        backend = _resolve_simulation_backend(simulation_backend)
        depths.sort()
    
        max_depth = depths[-1]
        num_depths = len(depths)
        num_cliff = len(experiments)
    
        data_table_shape = (num_cliff, num_depths, shots)
        check_table_shape = (num_cliff, num_depths, shots, max_depth + 1)
        measurement_record = np.zeros(data_table_shape, dtype=np.int64)
        # 1 records a pass, -1 records a fail, 0 is a default value that by the
        # end of LRB means it was an unused slot for depth < depths[-1].
        stabilizer_check_record = np.zeros(check_table_shape, dtype=np.int64)
    
        start_ng = 0
        start_k = 0
    
        for ng in range(start_ng, num_cliff):
            #print(f'Computing Clifford sequence {ng}')
            for k in range(start_k, num_depths):
    
                c = experiments[ng][k]
                simulated_shots = shots + 1
    
                # Run the circuits over many shots
                NoiseModelUtils.ensure_noise_params(c)
    
                total_start = time.perf_counter()
                simulation_output, timing_metrics = (
                    LRBSimulationPipeline.simulate_circuit(
                        c,
                        shots=simulated_shots,
                        simulation_backend=backend,
                        return_metrics=True,
                    )
                )

                unpack_start = time.perf_counter()
                unpack_payload = (
                    simulation_output
                    if getattr(unpack_func, "accepts_sdim_output", False)
                    else (
                        LRBSimulationPipeline.
                        measurement_results_from_sdim_output(
                            simulation_output)
                    )
                )
    
                #print("Cliff sequence is " + str(ng))
                #print("Depth / file is " + str(k))
                # Unpack the measurements
                #print(f"From LRB, we're passing current depth as {depths[k]}")
                stab_checks, m_values = unpack_func(
                    unpack_payload, depths[k], simulated_shots)
                unpack_seconds = time.perf_counter() - unpack_start

                # Record in table. Detector-backed unpackers return NumPy
                # arrays and legacy unpackers return lists; np.asarray keeps
                # both paths compatible while allowing vectorized assignment.
                # SDIM/DEM frame simulation uses shot zero as a noiseless
                # reference; discard it so each checkpoint contributes only
                # noisy sampled shots to the statistics.
                record_start = time.perf_counter()
                m_values_array = np.asarray(m_values, dtype=np.int64)[1:]
                stab_checks_array = np.asarray(
                    stab_checks,
                    dtype=np.int64,
                )[1:]
                measurement_record[ng, k, :shots] = m_values_array
                stabilizer_check_record[
                    ng,
                    k,
                    :shots,
                    :stab_checks_array.shape[1],
                ] = stab_checks_array
                record_seconds = time.perf_counter() - record_start

                timing_metrics.update(_circuit_timing_context(c))
                timing_metrics.update({
                    "phase": timing_phase,
                    "event": "simulate_depth",
                    "probability_index": probability_index,
                    "probability": probability,
                    "batch_index": batch_index,
                    "batch_shots": shots,
                    "clifford_index": ng,
                    "depth_index": k,
                    "depth": depths[k],
                    "shots": shots,
                    "unpack_seconds": unpack_seconds,
                    "record_seconds": record_seconds,
                    "total_seconds": time.perf_counter() - total_start,
                    "notes": (
                        f"simulated_shots={simulated_shots};"
                        "reference_shot_dropped=1"
                    ),
                })
                _append_timing_metric(
                    partial_progress_folder,
                    timing_metrics,
                )
    
        return measurement_record, stabilizer_check_record
    
    
    # An interval of 0 introduces no extra checks
    @staticmethod
    def postselect_uniform_interval(stabilizer_check_record, depths, interval):
    
        """
        Postselect uniform interval.
        
        Args:
            stabilizer_check_record (Any): Input argument.
            depths (Any): Input argument.
            interval (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        decision_table = np.zeros(stabilizer_check_record.shape[0:3],
                                  dtype=np.int64)
        num_cliff = stabilizer_check_record.shape[0]
        num_depths = stabilizer_check_record.shape[1]
        shots = stabilizer_check_record.shape[2]
    
        # make loop more general with nditer multi indices and slicing
        for ng in range(num_cliff):
            for k in range(num_depths):
                for s in range(shots):
    
                    # Terminal Stabilizer Check (after inversion step)
                    stab_pass = stabilizer_check_record[ng, k, s, depths[k]]
    
                    # Introduce extra checks only when depth is high enough.
                    # enough for the interval!
                    if depths[k] >= interval and interval > 0:
                        num_checks = depths[k] // interval
    
                        # intermediate checks
                        for j in range(1, num_checks +
                                       1):  # include the last forward check
                            idx = j * interval - 1  # forward segment index
                            # never touch the terminal slot
                            if idx < depths[k]:
                                stab_pass *= (
                                    stabilizer_check_record[ng, k, s, idx]
                                )
    
                    decision_table[ng, k, s] = stab_pass
    
        return decision_table
    
    
    # Number of checks 0 introduces no extra checks (obvious)
    @staticmethod
    def postselect_constant_number(
        stabilizer_check_record,
        depths,
        num_checks,
    ):
    
        """
        Postselect constant number.
        
        Args:
            stabilizer_check_record (Any): Input argument.
            depths (Any): Input argument.
            num_checks (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        decision_table = np.zeros(stabilizer_check_record.shape[0:3],
                                  dtype=np.int64)
        num_cliff = stabilizer_check_record.shape[0]
        num_depths = stabilizer_check_record.shape[1]
        shots = stabilizer_check_record.shape[2]
    
        # make loop more general with nditer multi indices and slicing
        for ng in range(num_cliff):
            for k in range(num_depths):
                for s in range(shots):
    
                    # Terminal Stabilizer Check (after inversion step)
                    stab_pass = stabilizer_check_record[ng, k, s, depths[k]]
    
                    # Do extra checks only if we can accomodate them
                    if num_checks > 0 and num_checks <= depths[k]:
                        interval = depths[k] // num_checks
    
                        # intermediate checks
                        for j in range(1, num_checks):
                            stab_pass = (
                                stab_pass
                                * stabilizer_check_record[
                                    ng, k, s, (j * interval) - 1
                                ]
                            )
    
                    decision_table[ng, k, s] = stab_pass
    
        return decision_table
    
    
    @staticmethod
    def postselect_record(measurement_record, decision_table):
    
        """
        Postselect record.
        
        Args:
            measurement_record (Any): Input argument.
            decision_table (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        out = measurement_record.copy()
        out[decision_table == 0] = -1
        return out
    
    
    @staticmethod
    def RB(
            experiments,
            depths: list[int],
            shots: int,
            simulation_backend: str | None = None,
            partial_progress_folder: str | None = None,
            timing_phase: str = "RB",
            probability_index: int | None = None,
            probability: float | None = None,
            batch_index: int | None = None):
        # 2D circuit table: first index is Clifford index, second is
        # experiment index, and entry i uses depth depths[i].
    
        """
        RB.
        
        Args:
            experiments (Any): Input argument.
            depths (Any): Input argument.
            shots (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        _require_sdim_runtime()
        backend = _resolve_simulation_backend(simulation_backend)
        depths.sort()
        num_depths = len(depths)
        num_cliff = len(experiments)
        data_table_shape = (num_cliff, num_depths, shots)
        measurement_record = np.zeros(data_table_shape, dtype=np.int64)
    
        for ng in range(num_cliff):
            for k in range(num_depths):
                c = experiments[ng][k]
                simulated_shots = shots + 1
                # Run the circuits over many shots
                NoiseModelUtils.ensure_noise_params(c)
    
                total_start = time.perf_counter()
                simulation_output, timing_metrics = (
                    LRBSimulationPipeline.simulate_circuit(
                        c,
                        shots=simulated_shots,
                        simulation_backend=backend,
                        return_metrics=True,
                    )
                )
                process_start = time.perf_counter()
                compact_results = (
                    LRBSimulationPipeline.
                    compact_detector_results_from_sdim_output(
                        simulation_output)
                )
                results = (
                    LRBSimulationPipeline.
                    measurement_results_from_sdim_output(simulation_output)
                )

                # New generated RB circuits contain one logical observable for
                # the terminal wire-zero measurement. Reconstruct all shots
                # from the reference shot plus SDIM's vectorized frame shifts
                # when that compact payload is present; older circuits still
                # fall back to the raw measurement tensor.
                if RB_LOGICAL_OBSERVABLE_LABEL in compact_results["logicals"]:
                    reference_value = int(results[0][0][0].measurement_value)
                    measurement_record[ng, k, :shots] = (
                        LRBSimulationPipeline.
                        detector_values_from_reference(
                            reference_value=reference_value,
                            detector_shifts=(
                                compact_results["logicals"][
                                    RB_LOGICAL_OBSERVABLE_LABEL]),
                            shots=simulated_shots,
                            dimension=c.dimension,
                        )[1:]
                    )
                else:
                    for s in range(shots):
                        measurement_record[ng, k, s] = (
                            results[0][0][s + 1].measurement_value)
                process_seconds = time.perf_counter() - process_start

                timing_metrics.update(_circuit_timing_context(c))
                timing_metrics.update({
                    "phase": timing_phase,
                    "event": "simulate_depth",
                    "probability_index": probability_index,
                    "probability": probability,
                    "batch_index": batch_index,
                    "batch_shots": shots,
                    "clifford_index": ng,
                    "depth_index": k,
                    "depth": depths[k],
                    "shots": shots,
                    "process_seconds": process_seconds,
                    "total_seconds": time.perf_counter() - total_start,
                    "notes": (
                        f"simulated_shots={simulated_shots};"
                        "reference_shot_dropped=1"
                    ),
                })
                _append_timing_metric(
                    partial_progress_folder,
                    timing_metrics,
                )
    
        return measurement_record
    
    
    @staticmethod
    def write_raw_data(filename, measurement_record, stabilizer_check_record):
        """
        Write raw data.
        
        Args:
            filename (Any): Input argument.
            measurement_record (Any): Input argument.
            stabilizer_check_record (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        np.savez(filename,
                 measurement_record=measurement_record,
                 stabilizer_check_record=stabilizer_check_record)
    
    
    @staticmethod
    def read_raw_data(filename):
        """
        Read raw data.
        
        Args:
            filename (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        f = filename
        if os.path.exists(f):
            raw_data = np.load(f)
            return raw_data['measurement_record'], raw_data[
                'stabilizer_check_record']
    
        return None
    
    
    # Document everything shortly after finishing
    # This function unpacks the 3D data generated above into fidelities in a
    # 2D array.
    @staticmethod
    def extract_statistics(measurement_record, dimension: int | None = None):
    
        """
        Extract statistics.
        
        Args:
            measurement_record (Any): Input argument.
            dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        shape = (measurement_record.shape[1], measurement_record.shape[0])
        shots = measurement_record.shape[2]
        fidelities = np.zeros(shape, dtype=np.float64)
        rejected_runs = np.zeros(shape, dtype=np.float64)
    
        valid_data = measurement_record[measurement_record >= 0]
        if dimension is None:
            dimension = int(valid_data.max()) + 1 if valid_data.size else 1
        elif valid_data.size and int(valid_data.max()) >= dimension:
            raise ValueError(
                "Measurement record contains outcomes outside the provided "
                "dimension."
            )
    
        omega = np.exp((2 * np.pi * (1j)) / dimension)
        phase_factors = omega**np.arange(dimension)
    
        fidelity_stats = []
        rejected_stats = []
    
        for k in range(shape[0]):
    
            #print(f"Calculating fidelities for depth: {depths[k]}")
    
            # Calculate fidelities over all Clifford runs at this depth
            for ng in range(shape[1]):
                data = measurement_record[ng, k]
                accepted = data[data >= 0]
                rejected_runs[k, ng] = (shots - accepted.size) / shots
    
                if accepted.size == 0:
                    fidelities[k, ng] = 0.0
                    continue
    
                tallies = np.bincount(accepted, minlength=dimension)
                fidelity = np.absolute(np.dot(tallies, phase_factors) /
                                       accepted.size)
    
                fidelities[k, ng] = fidelity
    
            # Calculate stats at given depth
            fidelity_stats.append({})
            rejected_stats.append({})
    
            #print(f"fidelity is: {fidelities[k]}")
    
            fidelity_stats[-1]['mean'] = np.mean(fidelities[k], axis=(0))
            rejected_stats[-1]['mean'] = np.mean(rejected_runs[k], axis=(0))
    
            if shape[1] == 1:
                fidelity_stats[-1]['std'] = None
                rejected_stats[-1]['std'] = None
            else:
                fidelity_stats[-1]['std'] = np.std(fidelities[k], axis=(0))
                rejected_stats[-1]['std'] = np.std(rejected_runs[k], axis=(0))
    
        return fidelity_stats, fidelities, rejected_stats, rejected_runs
    
    
    @staticmethod
    def extract_lrb_counts(measurement_record, dimension: int = 3):
    
        """
        Extract lrb counts.
        
        Args:
            measurement_record (Any): Input argument.
            dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        shape = (measurement_record.shape[1], measurement_record.shape[0],
                 dimension)
        shots = measurement_record.shape[2]
        m_counts = np.zeros(shape, dtype=np.int32)
        rejected_runs = np.zeros(shape[:2], dtype=np.int32)
    
        for k in range(shape[0]):
    
            #print(f"Calculating fidelities for depth: {depths[k]}")
    
            # Calculate fidelities over all Clifford runs at this depth
            for ng in range(shape[1]):
                data = measurement_record[ng, k]
                #tallies = [0 for j in range(dimension)]
                tallies = m_counts[k, ng]
    
                for i in range(shots):
                    if data[i] == -1:
                        rejected_runs[k, ng] += 1
                    else:
                        tallies[data[i]] += 1
    
                # Fidelity calculation
                # num_accepted_runs = sum(tallies)
                # omega = np.exp( (2 * np.pi * (1j)) / dimension)
                # normalized_tallies = [
                #     tallies[i] / num_accepted_runs for i in range(dimension)
                # ]
                # fidelity_summands = [
                #     (omega ** i) * normalized_tallies[i]
                #     for i in range(dimension)
                # ]
                # fidelity = np.absolute(sum(fidelity_summands))
    
        return m_counts, rejected_runs
    
    
    @staticmethod
    def lrb_counts_to_statistics(measurement_record,
                                 rejected_runs,
                                 BATCH_SIZE,
                                 num_unfiltered_shots,
                                 filter_trivial_shots: bool = False,
                                 dimension: int = 3):
    
        """
        Lrb counts to statistics.
        
        Args:
            measurement_record (Any): Input argument.
            rejected_runs (Any): Input argument.
            BATCH_SIZE (Any): Input argument.
            num_unfiltered_shots (Any): Input argument.
            filter_trivial_shots (Any): Input argument.
            dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        shape = (measurement_record.shape[0], measurement_record.shape[1])
    
        #raw_shots = measurement_record.shape[2]
        num_trivial_shots = num_unfiltered_shots // BATCH_SIZE
        offset = 0 if (num_unfiltered_shots % BATCH_SIZE) == 0 else 1
        num_trivial_shots += offset
        shots = (
            num_unfiltered_shots if not filter_trivial_shots else
            num_unfiltered_shots - num_trivial_shots
        )
    
        fidelities = np.zeros(shape, dtype=np.float64)
    
        fidelity_stats = []
        rejected_stats = []
    
        for k in range(shape[0]):
    
            #print(f"Calculating fidelities for depth: {depths[k]}")
    
            # Calculate fidelities over all Clifford runs at this depth
            for ng in range(shape[1]):
                tallies = measurement_record[k, ng]
    
                if filter_trivial_shots:
                    tallies[0] -= num_trivial_shots
                    rejected_runs[k, ng] += num_trivial_shots
                    assert (tallies[0] >= 0)
    
                # sanity check filtered data
                # print(f"tallies {tallies} and sum {sum(tallies)}")
                # print(f"rejected runs {rejected_runs[k, ng]}")
                assert (sum(tallies) +
                        rejected_runs[k, ng] == num_unfiltered_shots)
    
                # Fidelity calculation
                num_accepted_runs = sum(tallies)
                omega = np.exp((2 * np.pi * (1j)) / dimension)
                normalized_tallies = [
                    tallies[i] / num_accepted_runs for i in range(dimension)
                ]
                fidelity_summands = [(omega**i) * normalized_tallies[i]
                                     for i in range(dimension)]
                fidelity = np.absolute(sum(fidelity_summands))
    
                fidelities[k, ng] = fidelity
    
            # Normalize by total submitted shots so rejected proportion stays
            # in [0, 1] even when trivial shots are folded into rejection.
            normalized_rejected_runs = rejected_runs / num_unfiltered_shots
            # Calculate stats at given depth
            fidelity_stats.append({})
            rejected_stats.append({})
    
            #print(f"fidelity is: {fidelities[k]}")
    
            fidelity_stats[-1]['mean'] = np.mean(fidelities[k], axis=(0))
            rejected_stats[-1]['mean'] = np.mean(normalized_rejected_runs[k],
                                                 axis=(0))
    
            if shape[1] == 1:
                fidelity_stats[-1]['std'] = None
                rejected_stats[-1]['std'] = None
            else:
                fidelity_stats[-1]['std'] = np.std(fidelities[k], axis=(0))
                rejected_stats[-1]['std'] = np.std(
                    normalized_rejected_runs[k], axis=(0))
    
        return fidelity_stats, fidelities, rejected_stats, rejected_runs
    
    
    @staticmethod
    def write_stats(filename, prob, fidelity_stats, rejected_stats):
        """
        Write stats.
        
        Args:
            filename (Any): Input argument.
            prob (Any): Input argument.
            fidelity_stats (Any): Input argument.
            rejected_stats (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        parent = os.path.dirname(os.path.abspath(filename))
        os.makedirs(parent, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            dir=parent,
            prefix=f".{os.path.basename(filename)}.",
            suffix=".tmp",
            text=True,
        )
        try:
            csvfile = os.fdopen(descriptor, "w", newline="")
            len_fid = len(fidelity_stats)
            len_rej = len(rejected_stats)
            with csvfile:
                writer = csv.writer(csvfile)
                data = []
                data.append(["Probability", prob])
                data.append(["Fidelity averages"] +
                            [fidelity_stats[i]['mean']
                             for i in range(len_fid)])
                data.append(["Fidelity Standard Deviations"] +
                            [fidelity_stats[i]['std']
                             for i in range(len_fid)])
                data.append(["Rejected Runs"] +
                            [rejected_stats[i]['mean']
                             for i in range(len_rej)])
                data.append(["Rejected Standard Deviations"] +
                            [rejected_stats[i]['std']
                             for i in range(len_rej)])
                writer.writerows(data)
                csvfile.flush()
                os.fsync(csvfile.fileno())
            os.replace(temporary, filename)
            _fsync_directory(parent)
        except BaseException:
            if os.path.exists(temporary):
                os.unlink(temporary)
            raise
    
    
    @staticmethod
    def read_stats(filename):
        """
        Read stats.
        
        Args:
            filename (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        with open(filename, "r") as csvfile:
            reader = csv.reader(csvfile)
            # Format is above
            data = []
            for line in reader:
                data.append(line)
    
            prob = float(data[0][1])
    
            fidelity_stats = []
            rejected_stats = []
            len_data = len(data[1]) - 1
    
            for i in range(len_data):
                fidelity_stats.append({})
                rejected_stats.append({})
    
                fidelity_stats[-1]['mean'] = float(data[1][i + 1])
                fidelity_stats[-1]['std'] = float(data[2][i + 1])
    
                rejected_stats[-1]['mean'] = float(data[3][i + 1])
                rejected_stats[-1]['std'] = float(data[4][i + 1])
    
        return prob, fidelity_stats, rejected_stats
    
    
    @staticmethod
    def process_lrb_counts(measurement_results, stabilizer_record, depths,
                           stab_check_array, stab_checks_are_const, folder,
                           dimension: int = 3):
    
        """
        Process lrb counts.
        
        Args:
            measurement_results (Any): Input argument.
            stabilizer_record (Any): Input argument.
            depths (Any): Input argument.
            stab_check_array (Any): Input argument.
            stab_checks_are_const (Any): Input argument.
            folder (Any): Input argument.
            dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        shape = (measurement_results.shape[1], measurement_results.shape[0],
                 dimension)
    
        path_suffix = (
            "const_check_data/" if stab_checks_are_const else
            "unif_check_data/"
        )
        folder_path = folder + path_suffix
    
        if not os.path.exists(folder_path):
            os.mkdir(folder_path)
    
        for check in stab_check_array:
    
            if stab_checks_are_const:
                decision_table = (
                    LRBSimulationPipeline.postselect_constant_number(
                    stabilizer_check_record=stabilizer_record,
                    depths=depths,
                    num_checks=check
                    )
                )
            else:
                decision_table = (
                    LRBSimulationPipeline.postselect_uniform_interval(
                    stabilizer_check_record=stabilizer_record,
                    depths=depths,
                    interval=check
                    )
                )

            postselected_results = LRBSimulationPipeline.postselect_record(
                measurement_record=measurement_results,
                decision_table=decision_table)
            data_from_experiment, rejected_runs = (
                LRBSimulationPipeline.extract_lrb_counts(
                    measurement_record=postselected_results,
                    dimension=dimension))
    
            data_filename = folder_path + str(check) + ".npy"
            rejected_filename = folder_path + str(check) + "_rejected.npy"
    
            if not os.path.exists(data_filename):
                arr = np.zeros(shape, dtype=np.int32)
                np.save(data_filename, arr)
    
            if not os.path.exists(rejected_filename):
                arr = np.zeros(shape[:2], dtype=np.int32)
                np.save(rejected_filename, arr)
    
            counts_from_disk = np.load(data_filename)
            data_from_experiment = data_from_experiment + counts_from_disk
            np.save(data_filename, data_from_experiment)
    
            rejected_counts_from_disk = np.load(rejected_filename)
            rejected_runs = rejected_runs + rejected_counts_from_disk
            np.save(rejected_filename, rejected_runs)
    
            #print(f"Rejected Runs (TOTAL) is :\n\n{rejected_runs}")
    
        return 0
    
    
    @staticmethod
    def write_lrb_stats(read_directory, save_directory, prob, params,
                        BATCH_SIZE, num_unfiltered_shots,
                        filter_trivial_shots, dimension: int = 3):
    
        """
        Write lrb stats.
        
        Args:
            read_directory (Any): Input argument.
            save_directory (Any): Input argument.
            prob (Any): Input argument.
            params (Any): Input argument.
            BATCH_SIZE (Any): Input argument.
            num_unfiltered_shots (Any): Input argument.
            filter_trivial_shots (Any): Input argument.
            dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        for check in params:
            measurement_results = np.load(read_directory + str(check) + ".npy")
            rejected_runs = np.load(
                read_directory + str(check) + "_rejected.npy"
            )
            fidelity_stats, _, rejected_stats, _ = (
                LRBSimulationPipeline.lrb_counts_to_statistics(
                    measurement_results,
                    rejected_runs,
                    BATCH_SIZE,
                    num_unfiltered_shots,
                    filter_trivial_shots,
                    dimension=dimension,
                )
            )
            LRBSimulationPipeline.write_stats(
                save_directory + str(check) + ".csv",
                prob,
                fidelity_stats,
                rejected_stats,
            )
    
    
    @staticmethod
    def run_LRB_round(
        stab_checks_const,
        stab_checks_unif,
        BATCH_SIZE,
        error_prob,
        error_prob_ind,
        num_cliff_seq,
        depths,
        num_shots,
        filter_trivial_shots,
        LRB_experiment_folder_path,
        RB_experiment_folder_path,
        LRB_results_folder_path,
        RB_results_folder_path,
        partial_progress_folder_path,
        unpack_func=None,
        const0_unpack_func=None,
        logical_dimension: int = 3,
        LRB_const0_experiment_folder_path: str | None = None,
        simulation_backend: str | None = None,
        runtime_profile: str = "unspecified",
    ):
        """
        Run LRB round.
        
        Args:
            stab_checks_const (Any): Input argument.
            stab_checks_unif (Any): Input argument.
            BATCH_SIZE (Any): Input argument.
            error_prob (Any): Input argument.
            error_prob_ind (Any): Input argument.
            num_cliff_seq (Any): Input argument.
            depths (Any): Input argument.
            num_shots (Any): Input argument.
            filter_trivial_shots (Any): Input argument.
            LRB_experiment_folder_path (Any): Input argument.
            RB_experiment_folder_path (Any): Input argument.
            LRB_results_folder_path (Any): Input argument.
            RB_results_folder_path (Any): Input argument.
            partial_progress_folder_path (Any): Input argument.
            unpack_func (Any): Input argument.
            logical_dimension (Any): Input argument.
        
        Returns:
            object: Method output.
        
        Raises:
            ValueError: If supplied arguments violate this method's input
                assumptions.
        """
        _require_sdim_runtime()
        backend = _resolve_simulation_backend(simulation_backend)
        if unpack_func is None:
            raise ValueError("run_LRB_round requires an explicit unpack_func.")

        LRB_experiments = []
        LRB_const0_experiments = []
        RB_experiments = []
        num_depths = len(depths)
        const0_requested = 0 in stab_checks_const
        main_const_checks = [
            check for check in stab_checks_const if check != 0
        ]
        need_main_lrb = bool(main_const_checks or stab_checks_unif)
        if const0_requested and const0_unpack_func is None:
            raise ValueError(
                "const=0 processing requires an explicit const0_unpack_func."
            )
        if const0_requested and LRB_const0_experiment_folder_path is None:
            raise ValueError(
                "const=0 processing requires generated LRB_const0 circuits."
            )
        if (const0_requested
                and not os.path.exists(LRB_const0_experiment_folder_path)):
            raise ValueError(
                "const=0 processing requires generated LRB_const0 circuits "
                f"at {LRB_const0_experiment_folder_path}."
            )

        (
            checkpoint_store,
            input_fingerprint,
            circuit_folders,
        ) = open_lrb_checkpoint_store(
            partial_progress_folder_path=partial_progress_folder_path,
            num_shots=num_shots,
            batch_size=BATCH_SIZE,
            num_cliff_seq=num_cliff_seq,
            depths=depths,
            stab_checks_const=stab_checks_const,
            stab_checks_unif=stab_checks_unif,
            logical_dimension=logical_dimension,
            error_prob_ind=error_prob_ind,
            error_prob=error_prob,
            backend=backend,
            runtime_profile=runtime_profile,
            filter_trivial_shots=filter_trivial_shots,
            lrb_experiment_folder_path=LRB_experiment_folder_path,
            lrb_const0_experiment_folder_path=(
                LRB_const0_experiment_folder_path),
            rb_experiment_folder_path=RB_experiment_folder_path,
        )

        # Load only an atomically published checkpoint generation. Incomplete
        # copy-on-write transaction directories are never made current.
        shots_to_process = checkpoint_store.shots_remaining
        print(f"We have to process {shots_to_process} shots.")
        print(f"Simulation backend: {backend}")
    
        # Determine whether experiment is done
        if shots_to_process > 0:
    
            if need_main_lrb:
                load_start = time.perf_counter()
                # Otherwise read in all LRB experiment data
                for i in range(num_cliff_seq):
                    # make new list
                    LRB_experiments_from_single_sequence = []
                    LRB_cliff_path = LRB_experiment_folder_path + \
                        str(i) + '/' + str(error_prob_ind) + '/'

                    # iterate through depths
                    for j in range(num_depths):
                        LRB_single_experiment_path = (
                            LRB_cliff_path + str(j) + ".chp"
                        )
                        LRB_c = read_circuit(
                            os.path.abspath(LRB_single_experiment_path))
                        LRB_experiments_from_single_sequence.append(LRB_c)

                    LRB_experiments.append(LRB_experiments_from_single_sequence)
                load_seconds = time.perf_counter() - load_start
                _append_timing_metric(
                    partial_progress_folder_path,
                    {
                        "backend": backend,
                        "phase": "setup",
                        "event": "load_lrb_circuits",
                        "probability_index": error_prob_ind,
                        "probability": error_prob,
                        "load_seconds": load_seconds,
                        "total_seconds": load_seconds,
                        "notes": f"num_cliff={num_cliff_seq};"
                        f"num_depths={num_depths}",
                    },
                )

            if const0_requested:
                load_start = time.perf_counter()
                for i in range(num_cliff_seq):
                    const0_from_single_sequence = []
                    const0_cliff_path = (
                        LRB_const0_experiment_folder_path
                        + str(i) + '/' + str(error_prob_ind) + '/'
                    )

                    for j in range(num_depths):
                        const0_single_experiment_path = (
                            const0_cliff_path + str(j) + ".chp"
                        )
                        const0_c = read_circuit(
                            os.path.abspath(
                                const0_single_experiment_path))
                        const0_from_single_sequence.append(const0_c)

                    LRB_const0_experiments.append(
                        const0_from_single_sequence)
                load_seconds = time.perf_counter() - load_start
                _append_timing_metric(
                    partial_progress_folder_path,
                    {
                        "backend": backend,
                        "phase": "setup",
                        "event": "load_lrb_const0_circuits",
                        "probability_index": error_prob_ind,
                        "probability": error_prob,
                        "load_seconds": load_seconds,
                        "total_seconds": load_seconds,
                        "notes": f"num_cliff={num_cliff_seq};"
                        f"num_depths={num_depths}",
                    },
                )
    
            # Read in stabilizer check parameters
            # stab_checks_const = fetch_list(
            #     partial_progress_folder_path + "/check_const.txt"
            # )
            # stab_checks_unif = fetch_list(
            #     partial_progress_folder_path + "/check_unif.txt"
            # )
    
            # Run shots in batches of size BATCH_SIZE
            batch_index = (
                (num_shots - shots_to_process) // BATCH_SIZE
            )
            while shots_to_process > 0:
                if _circuit_input_fingerprint(
                    circuit_folders,
                    num_cliff_seq=num_cliff_seq,
                    probability_index=error_prob_ind,
                    num_depths=num_depths,
                ) != input_fingerprint:
                    raise RuntimeError(
                        "Circuit inputs changed during this probability-index "
                        "run; refusing to mix batches from different inputs."
                    )
    
                # Determine how many shots to run
                batch = (
                    BATCH_SIZE if shots_to_process > BATCH_SIZE else
                    shots_to_process
                )
                batch_index += 1
                shots_remaining_before = shots_to_process
                batch_progress_folder = checkpoint_store.prepare_batch(
                    shots_remaining_before=shots_remaining_before,
                    batch_index=batch_index,
                )
    
                # Run experiment
                print(
                    f"Resuming experiments for error probability {error_prob}"
                )
                start_time = time.time()
                batch_start = time.perf_counter()
                if need_main_lrb:
                    LRB_experiment_results = LRBSimulationPipeline.LRB(
                        experiments=LRB_experiments,
                        depths=depths,
                        shots=batch,
                        unpack_func=unpack_func,
                        partial_progress_folder=partial_progress_folder_path,
                        simulation_backend=backend,
                        timing_phase="LRB",
                        probability_index=error_prob_ind,
                        probability=error_prob,
                        batch_index=batch_index)
                else:
                    LRB_experiment_results = None
                end_time = time.time()
                batch_seconds = time.perf_counter() - batch_start
                print(f"Finished in {str(end_time - start_time)} seconds!")
                if need_main_lrb:
                    _append_timing_metric(
                        partial_progress_folder_path,
                        {
                            "backend": backend,
                            "phase": "LRB",
                            "event": "batch_total",
                            "probability_index": error_prob_ind,
                            "probability": error_prob,
                            "batch_index": batch_index,
                            "batch_shots": batch,
                            "shots_remaining_before": (
                                shots_remaining_before),
                            "total_seconds": batch_seconds,
                        },
                    )
    
                # Process partial results into progress files. Constant
                # checks with value > 0 are not terminal direct-data checks:
                # they are postselection views of the ordinary LRB stabilizer
                # record. For split-ancilla profiles, this means those const
                # checks use the split syndrome/relay ancilla convention.
                if need_main_lrb:
                    # const=0 is intentionally excluded from this list and is
                    # handled below with the direct terminal X-data circuit.
                    if main_const_checks:
                        process_start = time.perf_counter()
                        LRBSimulationPipeline.process_lrb_counts(
                            measurement_results=LRB_experiment_results[0],
                            stabilizer_record=LRB_experiment_results[1],
                            depths=depths,
                            stab_check_array=main_const_checks,
                            stab_checks_are_const=True,
                            folder=batch_progress_folder,
                            dimension=logical_dimension)
                        process_seconds = time.perf_counter() - process_start
                        _append_timing_metric(
                            partial_progress_folder_path,
                            {
                                "backend": backend,
                                "phase": "LRB",
                                "event": "process_const_checks",
                                "probability_index": error_prob_ind,
                                "probability": error_prob,
                                "batch_index": batch_index,
                                "batch_shots": batch,
                                "shots_remaining_before": (
                                    shots_remaining_before),
                                "process_seconds": process_seconds,
                                "total_seconds": process_seconds,
                                "notes": f"checks={main_const_checks}",
                            },
                        )
                    # Uniform checks retain the regular LRB circuits.
                    if stab_checks_unif:
                        process_start = time.perf_counter()
                        LRBSimulationPipeline.process_lrb_counts(
                            measurement_results=LRB_experiment_results[0],
                            stabilizer_record=LRB_experiment_results[1],
                            depths=depths,
                            stab_check_array=stab_checks_unif,
                            stab_checks_are_const=False,
                            folder=batch_progress_folder,
                            dimension=logical_dimension)
                        process_seconds = time.perf_counter() - process_start
                        _append_timing_metric(
                            partial_progress_folder_path,
                            {
                                "backend": backend,
                                "phase": "LRB",
                                "event": "process_unif_checks",
                                "probability_index": error_prob_ind,
                                "probability": error_prob,
                                "batch_index": batch_index,
                                "batch_shots": batch,
                                "shots_remaining_before": (
                                    shots_remaining_before),
                                "process_seconds": process_seconds,
                                "total_seconds": process_seconds,
                                "notes": f"checks={stab_checks_unif}",
                            },
                        )

                if const0_requested:
                    print("Running const=0 direct terminal X check...")
                    const0_start = time.perf_counter()
                    const0_results = LRBSimulationPipeline.LRB(
                        experiments=LRB_const0_experiments,
                        depths=depths,
                        shots=batch,
                        unpack_func=const0_unpack_func,
                        partial_progress_folder=partial_progress_folder_path,
                        simulation_backend=backend,
                        timing_phase="LRB_const0",
                        probability_index=error_prob_ind,
                        probability=error_prob,
                        batch_index=batch_index)
                    const0_seconds = time.perf_counter() - const0_start
                    _append_timing_metric(
                        partial_progress_folder_path,
                        {
                            "backend": backend,
                            "phase": "LRB_const0",
                            "event": "batch_total",
                            "probability_index": error_prob_ind,
                            "probability": error_prob,
                            "batch_index": batch_index,
                            "batch_shots": batch,
                            "shots_remaining_before": (
                                shots_remaining_before),
                            "total_seconds": const0_seconds,
                        },
                    )
                    process_start = time.perf_counter()
                    LRBSimulationPipeline.process_lrb_counts(
                        measurement_results=const0_results[0],
                        stabilizer_record=const0_results[1],
                        depths=depths,
                        stab_check_array=[0],
                        stab_checks_are_const=True,
                        folder=batch_progress_folder,
                        dimension=logical_dimension)
                    process_seconds = time.perf_counter() - process_start
                    _append_timing_metric(
                        partial_progress_folder_path,
                        {
                            "backend": backend,
                            "phase": "LRB_const0",
                            "event": "process_const0_check",
                            "probability_index": error_prob_ind,
                            "probability": error_prob,
                            "batch_index": batch_index,
                            "batch_shots": batch,
                            "shots_remaining_before": (
                                shots_remaining_before),
                            "process_seconds": process_seconds,
                            "total_seconds": process_seconds,
                            "notes": "checks=[0]",
                        },
                    )
    
                # Update shots processed
                shots_to_process -= batch
    
                write_start = time.perf_counter()
                checkpoint_store.commit_batch(
                    batch_progress_folder,
                    shots_remaining_after=shots_to_process,
                    batch_index=batch_index,
                )
                print(
                    f"Progress saved, need to process "
                    f"{shots_to_process} more shots."
                )
                write_seconds = time.perf_counter() - write_start
                _append_timing_metric(
                    partial_progress_folder_path,
                    {
                        "backend": backend,
                        "phase": "progress",
                        "event": "commit_checkpoint_generation",
                        "probability_index": error_prob_ind,
                        "probability": error_prob,
                        "batch_index": batch_index,
                        "batch_shots": batch,
                        "shots_remaining_before": shots_remaining_before,
                        "shots_remaining_after": shots_to_process,
                        "write_seconds": write_seconds,
                        "total_seconds": write_seconds,
                    },
                )

        if _circuit_input_fingerprint(
            circuit_folders,
            num_cliff_seq=num_cliff_seq,
            probability_index=error_prob_ind,
            num_depths=num_depths,
        ) != input_fingerprint:
            raise RuntimeError(
                "Circuit inputs changed before result generation; refusing "
                "to publish mixed-input artifacts."
            )
    
        # Calculate and write LRB stats
        const_save_dir = LRB_results_folder_path + \
            str(error_prob_ind) + "/const_check_data/"
        unif_save_dir = LRB_results_folder_path + \
            str(error_prob_ind) + "/unif_check_data/"
    
        for save_directory in (const_save_dir, unif_save_dir):
            if not os.path.exists(save_directory):
                os.makedirs(save_directory)
                _fsync_directory(
                    os.path.dirname(os.path.normpath(save_directory))
                )
    
        write_start = time.perf_counter()
        LRBSimulationPipeline.write_lrb_stats(
            checkpoint_store.current_data_folder + "const_check_data/",
            const_save_dir,
            error_prob, stab_checks_const, BATCH_SIZE, num_shots,
            filter_trivial_shots, dimension=logical_dimension)
        write_seconds = time.perf_counter() - write_start
        _append_timing_metric(
            partial_progress_folder_path,
            {
                "backend": backend,
                "phase": "LRB",
                "event": "write_const_stats",
                "probability_index": error_prob_ind,
                "probability": error_prob,
                "write_seconds": write_seconds,
                "total_seconds": write_seconds,
                "notes": f"checks={stab_checks_const}",
            },
        )

        write_start = time.perf_counter()
        LRBSimulationPipeline.write_lrb_stats(
            checkpoint_store.current_data_folder + "unif_check_data/",
            unif_save_dir,
            error_prob, stab_checks_unif, BATCH_SIZE, num_shots,
            filter_trivial_shots, dimension=logical_dimension)
        write_seconds = time.perf_counter() - write_start
        _append_timing_metric(
            partial_progress_folder_path,
            {
                "backend": backend,
                "phase": "LRB",
                "event": "write_unif_stats",
                "probability_index": error_prob_ind,
                "probability": error_prob,
                "write_seconds": write_seconds,
                "total_seconds": write_seconds,
                "notes": f"checks={stab_checks_unif}",
            },
        )
        print(
            "Wrote logical test data to the following directories:\n"
            f"{const_save_dir}\n"
            f"{unif_save_dir}"
        )
    
        # LRB_writefile = (
        #     LRB_results_folder_path + str(error_prob_ind) + '.csv'
        # )
        # LRB_raw_data_file = (
        #     LRB_results_folder_path + str(error_prob_ind)
        #     + '_raw_LRB_data.npz'
        # )
        # write_raw_data(
        #     LRB_raw_data_file,
        #     LRB_experiment_results[0],
        #     LRB_experiment_results[1],
        # )
        # print(f"Wrote logical test raw data  to {LRB_raw_data_file}")
    
        # Complete RB
        load_start = time.perf_counter()
        for i in range(num_cliff_seq):
    
            # make new list
            RB_experiments_from_single_sequence = []
            RB_cliff_path = RB_experiment_folder_path + \
                str(i) + '/' + str(error_prob_ind) + '/'
    
            # iterate through depths
            for j in range(num_depths):
                RB_single_experiment_path = RB_cliff_path + str(j) + ".chp"
                RB_c = read_circuit(os.path.abspath(RB_single_experiment_path))
                RB_experiments_from_single_sequence.append(RB_c)
    
            RB_experiments.append(RB_experiments_from_single_sequence)
        load_seconds = time.perf_counter() - load_start
        _append_timing_metric(
            partial_progress_folder_path,
            {
                "backend": backend,
                "phase": "RB",
                "event": "load_rb_circuits",
                "probability_index": error_prob_ind,
                "probability": error_prob,
                "load_seconds": load_seconds,
                "total_seconds": load_seconds,
                "notes": f"num_cliff={num_cliff_seq};"
                f"num_depths={num_depths}",
            },
        )
    
        print(f"Running physical circuit...")
        start_time = time.time()
        rb_start = time.perf_counter()
        RB_experiment_results = LRBSimulationPipeline.RB(
            experiments=RB_experiments,
            depths=depths,
            shots=NORMAL_RB_SHOTS,
            simulation_backend=backend,
            partial_progress_folder=partial_progress_folder_path,
            timing_phase="RB",
            probability_index=error_prob_ind,
            probability=error_prob,
            batch_index=0,
        )
        end_time = time.time()
        rb_seconds = time.perf_counter() - rb_start
        print(f"Finished in {str(end_time - start_time)} seconds!")
        _append_timing_metric(
            partial_progress_folder_path,
            {
                "backend": backend,
                "phase": "RB",
                "event": "batch_total",
                "probability_index": error_prob_ind,
                "probability": error_prob,
                "batch_index": 0,
                "batch_shots": NORMAL_RB_SHOTS,
                "total_seconds": rb_seconds,
            },
        )
    
        # Calculate and write RB stats
        RB_writefile = RB_results_folder_path + str(error_prob_ind) + '.csv'
        process_start = time.perf_counter()
        RB_f_stats, _, RB_r_stats, _ = (
            LRBSimulationPipeline.extract_statistics(
                measurement_record=RB_experiment_results,
                dimension=None,
            )
        )
        process_seconds = time.perf_counter() - process_start
        _append_timing_metric(
            partial_progress_folder_path,
            {
                "backend": backend,
                "phase": "RB",
                "event": "extract_statistics",
                "probability_index": error_prob_ind,
                "probability": error_prob,
                "batch_shots": NORMAL_RB_SHOTS,
                "process_seconds": process_seconds,
                "total_seconds": process_seconds,
            },
        )

        write_start = time.perf_counter()
        LRBSimulationPipeline.write_stats(filename=RB_writefile,
                                          prob=error_prob,
                                          fidelity_stats=RB_f_stats,
                                          rejected_stats=RB_r_stats)
        write_seconds = time.perf_counter() - write_start
        _append_timing_metric(
            partial_progress_folder_path,
            {
                "backend": backend,
                "phase": "RB",
                "event": "write_stats",
                "probability_index": error_prob_ind,
                "probability": error_prob,
                "batch_shots": NORMAL_RB_SHOTS,
                "write_seconds": write_seconds,
                "total_seconds": write_seconds,
                "notes": RB_writefile,
            },
        )
        print(f"Wrote physical test results to {RB_writefile}")
    
        return 0
    
# Shared default engine for scripts that prefer object-style calls.
DEFAULT_SIM_ENGINE = LRBSimulationEngine()
