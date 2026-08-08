from __future__ import annotations

import csv
import fcntl
import json
import os
from pathlib import Path

import numpy as np
import pytest

from lrb.lrb_simulation import (
    LRBBatchCheckpointStore,
    LRBSimulationPipeline,
    atomic_write_text,
)
from scripts.run_lrb_experiment import (
    LRBRunConfig,
    LRBRunCoordinator,
)


STORE_CONFIGURATION = {
    "num_shots": 20,
    "batch_size": 10,
    "num_cliff_seq": 2,
    "depths": (0, 2),
    "const_checks": (0, 1),
    "unif_checks": (1,),
    "dimension": 3,
    "probability_index": 4,
    "probability": 0.001,
    "backend": "dem",
    "input_fingerprint": "test-circuit-inputs-v1",
    "runtime_profile": "folded_qutrit",
    "filter_trivial_shots": False,
}


def make_store(progress_folder: Path, **overrides) -> LRBBatchCheckpointStore:
    configuration = dict(STORE_CONFIGURATION)
    configuration.update(overrides)
    return LRBBatchCheckpointStore(
        str(progress_folder),
        **configuration,
    )


def populate_transaction(
    transaction_folder: str,
    *,
    committed_shots: int,
    corrupt_check: tuple[str, int] | None = None,
) -> None:
    shape = (
        len(STORE_CONFIGURATION["depths"]),
        STORE_CONFIGURATION["num_cliff_seq"],
        STORE_CONFIGURATION["dimension"],
    )
    rejected_shape = shape[:2]
    policies = (
        ("const_check_data", STORE_CONFIGURATION["const_checks"]),
        ("unif_check_data", STORE_CONFIGURATION["unif_checks"]),
    )
    for directory, checks in policies:
        directory_path = Path(transaction_folder, directory)
        directory_path.mkdir(parents=True, exist_ok=True)
        for check in checks:
            counts = np.zeros(shape, dtype=np.int64)
            counts[..., 0] = committed_shots
            rejected = np.zeros(rejected_shape, dtype=np.int64)
            if corrupt_check == (directory, check):
                counts[0, 0, 0] += 1
            np.save(directory_path / f"{check}.npy", counts)
            np.save(directory_path / f"{check}_rejected.npy", rejected)


def commit_one_batch(store: LRBBatchCheckpointStore) -> None:
    transaction = store.prepare_batch(
        shots_remaining_before=20,
        batch_index=1,
    )
    populate_transaction(transaction, committed_shots=10)
    store.commit_batch(
        transaction,
        shots_remaining_after=10,
        batch_index=1,
    )


def write_stats_csv(path: Path, probability: float = 0.001) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = (
        ("Probability", probability),
        ("Fidelity averages", 1.0, 0.9),
        ("Fidelity Standard Deviations", 0.0, 0.01),
        ("Rejected Runs", 0.0, 0.1),
        ("Rejected Standard Deviations", 0.0, 0.01),
    )
    with path.open("w", newline="") as result_file:
        csv.writer(result_file).writerows(rows)


def test_initial_checkpoint_and_committed_restart(tmp_path: Path) -> None:
    progress = tmp_path / "progress"
    store = make_store(progress)

    assert store.shots_remaining == 20
    assert (progress / "shots_processed.txt").is_symlink()
    assert (progress / "const_check_data").is_symlink()
    assert (progress / "unif_check_data").is_symlink()

    commit_one_batch(store)
    assert store.shots_remaining == 10

    restarted = make_store(progress)
    assert restarted.shots_remaining == 10
    restarted.validate_current_generation()
    counts = np.load(
        progress / "const_check_data" / "0.npy",
        allow_pickle=False,
    )
    assert np.all(counts.sum(axis=2) == 10)


def test_multiple_commits_include_a_partial_final_batch(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress"
    store = make_store(progress, num_shots=10, batch_size=4)
    remaining = 10
    for batch_index, committed_shots in enumerate((4, 8, 10), start=1):
        transaction = store.prepare_batch(
            shots_remaining_before=remaining,
            batch_index=batch_index,
        )
        populate_transaction(
            transaction,
            committed_shots=committed_shots,
        )
        remaining = 10 - committed_shots
        store.commit_batch(
            transaction,
            shots_remaining_after=remaining,
            batch_index=batch_index,
        )
    assert store.shots_remaining == 0
    manifest = json.loads(
        Path(store.current_generation, "manifest.json").read_text()
    )
    assert manifest["last_batch_index"] == 3
    make_store(
        progress,
        num_shots=10,
        batch_size=4,
    ).validate_current_generation()


def test_unpublished_transaction_is_discarded_on_restart(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress"
    store = make_store(progress)
    transaction = store.prepare_batch(
        shots_remaining_before=20,
        batch_index=1,
    )
    populate_transaction(transaction, committed_shots=10)

    restarted = make_store(progress)
    assert restarted.shots_remaining == 20
    assert not list(
        (progress / ".checkpoint_generations").glob(".txn-*")
    )


@pytest.mark.parametrize(
    ("failpoint", "expected_remaining"),
    (
        ("after_staged_metadata", 20),
        ("after_generation_rename", 20),
        ("before_current_replace", 20),
        ("after_current_replace", 10),
    ),
)
def test_commit_publication_failpoints_are_restart_safe(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failpoint: str,
    expected_remaining: int,
) -> None:
    progress = tmp_path / failpoint
    store = make_store(progress)
    transaction = store.prepare_batch(
        shots_remaining_before=20,
        batch_index=1,
    )
    populate_transaction(transaction, committed_shots=10)
    monkeypatch.setenv("_LRB_TEST_CHECKPOINT_FAILPOINT", failpoint)
    with pytest.raises(RuntimeError, match="Injected checkpoint failure"):
        store.commit_batch(
            transaction,
            shots_remaining_after=10,
            batch_index=1,
        )
    monkeypatch.delenv("_LRB_TEST_CHECKPOINT_FAILPOINT")

    restarted = make_store(progress)
    assert restarted.shots_remaining == expected_remaining
    restarted.validate_current_generation()


def test_prepare_failpoint_leaves_old_generation_authoritative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    progress = tmp_path / "progress"
    store = make_store(progress)
    monkeypatch.setenv(
        "_LRB_TEST_CHECKPOINT_FAILPOINT",
        "after_prepare_copy",
    )
    with pytest.raises(RuntimeError, match="after_prepare_copy"):
        store.prepare_batch(
            shots_remaining_before=20,
            batch_index=1,
        )
    monkeypatch.delenv("_LRB_TEST_CHECKPOINT_FAILPOINT")

    assert make_store(progress).shots_remaining == 20


def test_invalid_staged_totals_cannot_be_published(tmp_path: Path) -> None:
    store = make_store(tmp_path / "progress")
    transaction = store.prepare_batch(
        shots_remaining_before=20,
        batch_index=1,
    )
    populate_transaction(
        transaction,
        committed_shots=10,
        corrupt_check=("const_check_data", 1),
    )

    with pytest.raises(RuntimeError, match="expected exactly 10 per cell"):
        store.commit_batch(
            transaction,
            shots_remaining_after=10,
            batch_index=1,
        )
    assert store.shots_remaining == 20


def test_truncated_or_negative_checkpoint_is_rejected(tmp_path: Path) -> None:
    progress = tmp_path / "progress"
    store = make_store(progress)
    commit_one_batch(store)

    count_path = Path(
        store.current_generation,
        "const_check_data",
        "0.npy",
    )
    count_path.write_bytes(b"not-a-numpy-file")
    with pytest.raises(Exception):
        make_store(progress)

    count_path.unlink()
    negative = np.zeros((2, 2, 3), dtype=np.int64)
    negative[..., 0] = 10
    negative[0, 0, 0] = -1
    np.save(count_path, negative)
    with pytest.raises(RuntimeError, match="negative values"):
        make_store(progress)


def test_configuration_drift_is_rejected(tmp_path: Path) -> None:
    progress = tmp_path / "progress"
    make_store(progress)
    with pytest.raises(RuntimeError, match="fingerprint changed"):
        make_store(progress, batch_size=5)


def test_manifest_configuration_tampering_is_rejected(tmp_path: Path) -> None:
    progress = tmp_path / "progress"
    store = make_store(progress)
    manifest_path = Path(store.current_generation, "manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["configuration"]["backend"] = "tampered"
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(RuntimeError, match="modified or corrupted"):
        make_store(progress)


def test_invalid_shot_transition_cannot_replace_current(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress"
    store = make_store(progress)
    transaction = store.prepare_batch(
        shots_remaining_before=20,
        batch_index=1,
    )
    populate_transaction(transaction, committed_shots=21)

    with pytest.raises(RuntimeError, match="Invalid checkpoint transition"):
        store.commit_batch(
            transaction,
            shots_remaining_after=-1,
            batch_index=1,
        )
    assert store.shots_remaining == 20
    make_store(progress).validate_current_generation()


def test_partial_legacy_checkpoint_is_rejected(tmp_path: Path) -> None:
    progress = tmp_path / "progress"
    count_directory = progress / "const_check_data"
    count_directory.mkdir(parents=True)
    atomic_write_text(str(progress / "shots_processed.txt"), "20")
    np.save(count_directory / "0.npy", np.zeros((2, 2, 3), dtype=np.int64))

    with pytest.raises(RuntimeError, match="reports zero committed shots"):
        make_store(progress)


def test_committed_legacy_checkpoint_without_provenance_is_rejected(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress"
    progress.mkdir()
    atomic_write_text(str(progress / "shots_processed.txt"), "10")
    populate_transaction(str(progress), committed_shots=10)

    with pytest.raises(RuntimeError, match="no recorded source"):
        make_store(progress)


def test_exact_result_paths_and_csv_schema_are_validated(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress"
    lrb_results = tmp_path / "results" / "LRB"
    rb_results = tmp_path / "results" / "RB"
    progress.mkdir()
    atomic_write_text(str(progress / "shots_processed.txt"), "0")
    for check in (0, 1):
        write_stats_csv(lrb_results / "4" / "const_check_data" / f"{check}.csv")
    write_stats_csv(lrb_results / "4" / "unif_check_data" / "1.csv")
    write_stats_csv(rb_results / "4.csv")

    LRBRunCoordinator._validate_completed_artifacts(
        stab_checks_const=(0, 1),
        stab_checks_unif=(1,),
        probability=0.001,
        error_prob_ind=4,
        depths=(0, 2),
        lrb_results_folder_path=str(lrb_results),
        rb_results_folder_path=str(rb_results),
        partial_progress_folder_path=str(progress),
    )

    (lrb_results / "4" / "const_check_data" / "1.csv").unlink()
    write_stats_csv(lrb_results / "4" / "const_check_data" / "999.csv")
    with pytest.raises(RuntimeError, match="missing=.*1.csv.*extra=.*999.csv"):
        LRBRunCoordinator._validate_completed_artifacts(
            stab_checks_const=(0, 1),
            stab_checks_unif=(1,),
            probability=0.001,
            error_prob_ind=4,
            depths=(0, 2),
            lrb_results_folder_path=str(lrb_results),
            rb_results_folder_path=str(rb_results),
            partial_progress_folder_path=str(progress),
        )


def test_completion_manifest_detects_structurally_valid_tampering(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress"
    lrb_results = tmp_path / "results" / "LRB"
    rb_results = tmp_path / "results" / "RB"
    store = make_store(progress)
    commit_one_batch(store)
    transaction = store.prepare_batch(
        shots_remaining_before=10,
        batch_index=2,
    )
    populate_transaction(transaction, committed_shots=20)
    store.commit_batch(
        transaction,
        shots_remaining_after=0,
        batch_index=2,
    )
    for check in (0, 1):
        write_stats_csv(lrb_results / "4" / "const_check_data" / f"{check}.csv")
    write_stats_csv(lrb_results / "4" / "unif_check_data" / "1.csv")
    write_stats_csv(rb_results / "4.csv")
    manifest_arguments = {
        "stab_checks_const": (0, 1),
        "stab_checks_unif": (1,),
        "probability": 0.001,
        "error_prob_ind": 4,
        "lrb_results_folder_path": str(lrb_results),
        "rb_results_folder_path": str(rb_results),
        "partial_progress_folder_path": str(progress),
    }
    LRBRunCoordinator._write_completion_manifest(**manifest_arguments)
    LRBRunCoordinator._validate_completion_manifest(**manifest_arguments)

    tampered_path = lrb_results / "4" / "const_check_data" / "1.csv"
    tampered_path.write_text(
        tampered_path.read_text().replace(",0.9\n", ",0.8\n", 1)
    )
    with pytest.raises(RuntimeError, match="does not match"):
        LRBRunCoordinator._validate_completion_manifest(
            **manifest_arguments
        )


def test_atomic_stats_write_preserves_previous_file_on_replace_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result_path = tmp_path / "result.csv"
    result_path.write_text("previous-complete-file\n")
    real_replace = os.replace

    def fail_result_replacement(source: str, destination: str) -> None:
        if os.path.abspath(destination) == os.path.abspath(result_path):
            raise OSError("injected result replacement failure")
        real_replace(source, destination)

    monkeypatch.setattr(os, "replace", fail_result_replacement)
    stats = ({"mean": 1.0, "std": 0.0},)
    with pytest.raises(OSError, match="injected result replacement"):
        LRBSimulationPipeline.write_stats(
            str(result_path),
            0.001,
            stats,
            stats,
        )
    assert result_path.read_text() == "previous-complete-file\n"
    assert not list(tmp_path.glob(".result.csv.*.tmp"))


def test_probability_worker_lock_rejects_concurrent_owner(
    tmp_path: Path,
) -> None:
    progress = tmp_path / "progress"
    progress.mkdir()
    lock_path = progress / ".worker.lock"
    with lock_path.open("a+") as owner:
        fcntl.flock(owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        coordinator = LRBRunCoordinator(
            simulation_engine=None,
            config=LRBRunConfig(),
        )
        with pytest.raises(RuntimeError, match="Another worker already owns"):
            coordinator.compute_lrb(
                stab_checks_const=(0, 1),
                stab_checks_unif=(1,),
                probabilities=(0.001,),
                error_prob_ind=0,
                num_cliff_seq=2,
                depths=(0, 2),
                num_shots=20,
                lrb_experiment_folder_path=str(tmp_path / "experiments/LRB"),
                lrb_const0_experiment_folder_path=str(
                    tmp_path / "experiments/LRB_const0"
                ),
                rb_experiment_folder_path=str(tmp_path / "experiments/RB"),
                lrb_results_folder_path=str(tmp_path / "results/LRB"),
                rb_results_folder_path=str(tmp_path / "results/RB"),
                progress_file_path=str(progress / "done.txt"),
                partial_progress_folder_path=str(progress),
            )
