"""Walk-forward split helpers for time-series model validation."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeasonWalkForwardFold:
    """Walk-forward fold over seasons with separate train/val/eval indices.

    This mirrors the pattern used in exploratory notebooks for season-based
    validation:

    - ``train_indices``: all rows from seasons strictly earlier than both the
      validation and evaluation seasons,
    - ``val_indices``: rows from the intermediate validation season used for
      early stopping or other training-time diagnostics,
    - ``eval_indices``: rows from the next season used exclusively for
      out-of-sample evaluation in grid search.

    The validation slice must not be used as the sole basis for ranking
    hyperparameters in :func:`~src.models.tuning.trainable_grid_search.run_trainable_grid_search_three_way`
    — only ``eval_indices`` should be passed to ``predict`` for ranking.
    """

    fold_id: int
    train_indices: np.ndarray
    val_indices: np.ndarray
    eval_indices: np.ndarray


def make_season_walk_forward_splits(
    df: pd.DataFrame,
    *,
    season_col: str = "season",
    seasons_order: Sequence[str] | None = None,
    strict: bool = False,
) -> list[SeasonWalkForwardFold]:
    """Create season-based walk-forward splits with train/val/eval indices.

    The intended usage mirrors the XGBoost Poisson prototype notebook:

    - the input ``df`` should already be filtered to historical seasons
      (any holdout like ``\"current\"`` should be removed beforehand),
    - ``seasons_order`` controls the chronological order of seasons used for
      walk-forward validation,
    - for each step we use:
        * train = all seasons strictly earlier than both val and eval,
        * val   = the season immediately preceding eval (for early stopping),
        * eval  = the current season (for out-of-sample metrics).

    Parameters
    ----------
    df:
        Input DataFrame containing a season column.
    season_col:
        Name of the column with season labels. Default: ``\"season\"``.
    seasons_order:
        Explicit ordered list of seasons to use for walk-forward. If omitted,
        the order is inferred from the unique values in ``season_col`` in
        **appearance order** (row order in ``df``). For chronological walk-forward
        you should always pass ``seasons_order`` explicitly.
    strict:
        If ``True``, raise when any fold would have an empty train, val, or eval
        slice instead of skipping that fold. If ``False`` (default), degenerate
        folds are skipped with a warning.
    """
    if season_col not in df.columns:
        raise ValueError(f"Missing season column: '{season_col}'.")

    if seasons_order is None:
        # Appearance order follows df row order; callers needing chronology
        # should pass seasons_order explicitly.
        seasons_order = list(dict.fromkeys(df[season_col].tolist()))

    if len(seasons_order) < 3:
        raise ValueError(
            "At least three seasons are required for season-based "
            "train/val/eval walk-forward splits."
        )

    # Validate that all requested seasons are present in the data.
    available = set(df[season_col].unique())
    missing = [s for s in seasons_order if s not in available]
    if missing:
        raise ValueError(
            "The following seasons from 'seasons_order' are not present in the "
            f"DataFrame: {missing}."
        )

    season_values = df[season_col].to_numpy()
    folds: list[SeasonWalkForwardFold] = []

    # We start from index 2 to ensure we have at least:
    # - one or more training seasons,
    # - one validation season,
    # - one evaluation season.
    for eval_idx in range(2, len(seasons_order)):
        train_seasons = seasons_order[: eval_idx - 1]
        val_season = seasons_order[eval_idx - 1]
        eval_season = seasons_order[eval_idx]

        train_mask = np.isin(season_values, train_seasons)
        val_mask = season_values == val_season
        eval_mask = season_values == eval_season

        train_indices = np.flatnonzero(train_mask).astype(int)
        val_indices = np.flatnonzero(val_mask).astype(int)
        eval_indices = np.flatnonzero(eval_mask).astype(int)

        if train_indices.size == 0 or val_indices.size == 0 or eval_indices.size == 0:
            if strict:
                raise ValueError(
                    "Season walk-forward fold has empty split: "
                    f"val_season={val_season!r}, eval_season={eval_season!r}, "
                    f"train_n={train_indices.size}, val_n={val_indices.size}, "
                    f"eval_n={eval_indices.size}."
                )
            warnings.warn(
                f"Skipping degenerate season fold (val={val_season!r}, eval={eval_season!r}): "
                f"empty train/val/eval slice.",
                stacklevel=2,
            )
            continue

        folds.append(
            SeasonWalkForwardFold(
                fold_id=eval_idx - 1,
                train_indices=train_indices,
                val_indices=val_indices,
                eval_indices=eval_indices,
            )
        )

    if not folds:
        raise ValueError(
            "No valid season-based folds could be constructed. "
            "Check that each season in 'seasons_order' has at least one row."
        )

    return folds
