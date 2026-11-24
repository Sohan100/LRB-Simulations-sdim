"""Z_3 linear algebra and base-3 digit indexing used by the chain decoder."""

from __future__ import annotations

import numpy as np

BASE = 3


def digits_to_index(digits: np.ndarray) -> np.ndarray:
    """
    Convert base-3 digit vectors to integer indices (digit 0 least significant).

    Args:
        digits (np.ndarray): Integer array whose last axis holds the digits.

    Returns:
        np.ndarray: Integer indices with the last axis removed.
    """
    digits = np.asarray(digits, dtype=np.int64) % BASE
    powers = BASE ** np.arange(digits.shape[-1], dtype=np.int64)
    return digits @ powers


def index_to_digits(index: np.ndarray | int, width: int) -> np.ndarray:
    """
    Convert integer indices to base-3 digit vectors (digit 0 least significant).

    Args:
        index (np.ndarray | int): Integer index or array of indices.
        width (int): Number of digits to produce.

    Returns:
        np.ndarray: Digits with a new trailing axis of length ``width``.
    """
    index = np.asarray(index, dtype=np.int64)
    out = np.empty(index.shape + (width,), dtype=np.int64)
    rest = index.copy()
    for position in range(width):
        out[..., position] = rest % BASE
        rest //= BASE
    return out


def subtraction_table(width: int) -> np.ndarray:
    """
    Build ``table[x, y] = index(digits(x) - digits(y) mod 3)``.

    Args:
        width (int): Number of base-3 digits per index.

    Returns:
        np.ndarray: ``(3**width, 3**width)`` integer table.
    """
    size = BASE**width
    digits = index_to_digits(np.arange(size), width)
    diff = (digits[:, None, :] - digits[None, :, :]) % BASE
    return digits_to_index(diff)


def solve_linear_map(x_rows: np.ndarray, y_rows: np.ndarray) -> np.ndarray | None:
    """
    Find a matrix ``T`` over Z_3 with ``T @ x = y`` for every row pair.

    The rows of ``x_rows`` and ``y_rows`` are vectors ``x_i`` and ``y_i``. The
    solver row-reduces the augmented system ``[x | y]`` and returns ``None``
    when the pairs are inconsistent with any linear map. Directions not
    spanned by the ``x_i`` are set to zero, which is harmless because such
    directions never occur in the data used to build the map.

    Args:
        x_rows (np.ndarray): ``(n, k)`` input vectors.
        y_rows (np.ndarray): ``(n, k2)`` output vectors.

    Returns:
        np.ndarray | None: ``(k2, k)`` matrix over Z_3, or ``None``.
    """
    x_rows = np.asarray(x_rows, dtype=np.int64) % BASE
    y_rows = np.asarray(y_rows, dtype=np.int64) % BASE
    n_in = x_rows.shape[1]
    aug = np.concatenate([x_rows, y_rows], axis=1)
    pivot_row = 0
    pivots: list[int] = []
    for column in range(n_in):
        candidates = np.flatnonzero(aug[pivot_row:, column] % BASE)
        if candidates.size == 0:
            continue
        row = pivot_row + int(candidates[0])
        aug[[pivot_row, row]] = aug[[row, pivot_row]]
        inverse = 1 if aug[pivot_row, column] == 1 else 2
        aug[pivot_row] = (aug[pivot_row] * inverse) % BASE
        for other in range(aug.shape[0]):
            if other != pivot_row and aug[other, column]:
                aug[other] = (aug[other] - aug[other, column] * aug[pivot_row]) % BASE
        pivots.append(column)
        pivot_row += 1
        if pivot_row == aug.shape[0]:
            break
    # Rows with zero input part must have zero output part.
    if pivot_row < aug.shape[0] and np.any(aug[pivot_row:, n_in:] % BASE):
        return None
    transform = np.zeros((y_rows.shape[1], n_in), dtype=np.int64)
    for row_index, column in enumerate(pivots):
        transform[:, column] = aug[row_index, n_in:]
    if np.any((x_rows @ transform.T - y_rows) % BASE):
        return None
    return transform % BASE
