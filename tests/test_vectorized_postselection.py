"""Vectorized postselection matches the original per-shot loop semantics."""

from __future__ import annotations

import numpy as np
import pytest

from lrb.lrb_simulation import LRBSimulationPipeline


DEPTHS = [0, 1, 2, 4, 6, 10, 14, 18, 20, 22]
SHAPE = (3, len(DEPTHS), 50)


def loop_uniform_interval(record, depths, interval):
    """The original triple loop of ``postselect_uniform_interval``."""
    table = np.zeros(record.shape[0:3], dtype=np.int64)
    for ng in range(record.shape[0]):
        for k in range(record.shape[1]):
            for s in range(record.shape[2]):
                stab_pass = record[ng, k, s, depths[k]]
                if depths[k] >= interval and interval > 0:
                    for j in range(1, depths[k] // interval + 1):
                        idx = j * interval - 1
                        if idx < depths[k]:
                            stab_pass *= record[ng, k, s, idx]
                table[ng, k, s] = stab_pass
    return table


def loop_constant_number(record, depths, num_checks):
    """The original triple loop of ``postselect_constant_number``."""
    table = np.zeros(record.shape[0:3], dtype=np.int64)
    for ng in range(record.shape[0]):
        for k in range(record.shape[1]):
            for s in range(record.shape[2]):
                stab_pass = record[ng, k, s, depths[k]]
                if num_checks > 0 and num_checks <= depths[k]:
                    interval = depths[k] // num_checks
                    for j in range(1, num_checks):
                        stab_pass = stab_pass * record[
                            ng, k, s, (j * interval) - 1]
                table[ng, k, s] = stab_pass
    return table


def loop_extract_counts(record, dimension):
    """The original per-shot tally of ``extract_lrb_counts``."""
    shape = (record.shape[1], record.shape[0], dimension)
    counts = np.zeros(shape, dtype=np.int32)
    rejected = np.zeros(shape[:2], dtype=np.int32)
    for k in range(shape[0]):
        for ng in range(shape[1]):
            for value in record[ng, k]:
                if value == -1:
                    rejected[k, ng] += 1
                else:
                    counts[k, ng, value] += 1
    return counts, rejected


@pytest.mark.parametrize("values", [(0, 1), (-1, 0, 1)])
def test_decision_tables_match_loops(values):
    rng = np.random.default_rng(7)
    record = rng.choice(
        values, size=SHAPE + (max(DEPTHS) + 1,), p=None).astype(np.int64)
    record[..., :] = np.where(rng.random(record.shape) < 0.9, 1, record)
    for check in range(0, 24):
        np.testing.assert_array_equal(
            LRBSimulationPipeline.postselect_uniform_interval(
                record, DEPTHS, check),
            loop_uniform_interval(record, DEPTHS, check),
        )
        np.testing.assert_array_equal(
            LRBSimulationPipeline.postselect_constant_number(
                record, DEPTHS, check),
            loop_constant_number(record, DEPTHS, check),
        )


def test_extract_counts_matches_loop():
    rng = np.random.default_rng(11)
    record = rng.integers(-1, 3, size=SHAPE).astype(np.int64)
    counts, rejected = LRBSimulationPipeline.extract_lrb_counts(record, 3)
    expected_counts, expected_rejected = loop_extract_counts(record, 3)
    np.testing.assert_array_equal(counts, expected_counts)
    np.testing.assert_array_equal(rejected, expected_rejected)
