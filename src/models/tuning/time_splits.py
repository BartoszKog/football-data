"""Walk-forward split helpers for time-series model validation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesFold:
    """Single walk-forward fold with train/validation row indices."""

    fold_id: int
    train_indices: np.ndarray
    valid_indices: np.ndarray


def make_walk_forward_splits(
    df: pd.DataFrame,
    *,
    datetime_col: str,
    n_splits: int,
    min_train_size: int,
    valid_size: int,
) -> list[TimeSeriesFold]:
    """Create expanding-window walk-forward splits for a sorted time series.

    The function requires the DataFrame to be sorted ascending by
    ``datetime_col`` and returns row-position indices usable with ``iloc``.
    """
    _validate_time_split_inputs(
        df=df,
        datetime_col=datetime_col,
        n_splits=n_splits,
        min_train_size=min_train_size,
        valid_size=valid_size,
    )
    _ensure_datetime_column_sorted(df, datetime_col=datetime_col)

    folds: list[TimeSeriesFold] = []
    for split_idx in range(n_splits):
        train_end = min_train_size + (split_idx * valid_size)
        valid_start = train_end
        valid_end = valid_start + valid_size

        train_indices = np.arange(0, train_end, dtype=int)
        valid_indices = np.arange(valid_start, valid_end, dtype=int)
        folds.append(
            TimeSeriesFold(
                fold_id=split_idx,
                train_indices=train_indices,
                valid_indices=valid_indices,
            )
        )

    return folds


def _validate_time_split_inputs(
    *,
    df: pd.DataFrame,
    datetime_col: str,
    n_splits: int,
    min_train_size: int,
    valid_size: int,
) -> None:
    if datetime_col not in df.columns:
        raise ValueError(f"Missing datetime column: '{datetime_col}'.")
    if n_splits <= 0:
        raise ValueError("n_splits must be greater than 0.")
    if min_train_size <= 0:
        raise ValueError("min_train_size must be greater than 0.")
    if valid_size <= 0:
        raise ValueError("valid_size must be greater than 0.")

    needed_rows = min_train_size + (n_splits * valid_size)
    if len(df) < needed_rows:
        raise ValueError(
            "Not enough rows for requested walk-forward configuration: "
            f"len(df)={len(df)}, needed={needed_rows} "
            f"(min_train_size={min_train_size}, n_splits={n_splits}, "
            f"valid_size={valid_size})."
        )


def _ensure_datetime_column_sorted(df: pd.DataFrame, *, datetime_col: str) -> None:
    series = pd.to_datetime(df[datetime_col], errors="coerce")
    if series.isna().any():
        raise ValueError(f"Column '{datetime_col}' contains non-datetime values.")

    if not series.is_monotonic_increasing:
        raise ValueError(
            f"DataFrame must be sorted ascending by '{datetime_col}' before splitting."
        )
