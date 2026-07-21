from __future__ import annotations

import numpy as np


def mask_outliers(values, *, min_value: float | None = None, max_value: float | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return np.zeros(arr.shape, dtype=bool)

    finite_mask = np.isfinite(arr)
    if not np.any(finite_mask):
        return np.zeros(arr.shape, dtype=bool)

    arr_finite = arr[finite_mask]
    if arr_finite.size < 4:
        mask = np.ones(arr.shape, dtype=bool)
        mask[~finite_mask] = False
        return mask

    q1, q3 = np.percentile(arr_finite, [25, 75])
    iqr = q3 - q1
    if not np.isfinite(iqr) or iqr <= 0:
        mask_finite = np.ones(arr_finite.shape, dtype=bool)
    else:
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask_finite = (arr_finite >= lower) & (arr_finite <= upper)

    if min_value is not None:
        mask_finite &= arr_finite >= min_value
    if max_value is not None:
        mask_finite &= arr_finite <= max_value

    mask = np.zeros(arr.shape, dtype=bool)
    mask[finite_mask] = mask_finite
    return mask
