"""Grid search for models requiring train/validation fitting."""

from __future__ import annotations

import time
from typing import Any, Callable, Sequence, Mapping

import pandas as pd

from ..evaluation import evaluate_score_predictions
from ..interfaces import PredictiveModel, TrainablePredictiveModel
from .grid_search import (
    CacheMode,
    GridSearchResult,
    _build_cache_path,
    _parameter_combinations,
    _print_progress,
    _read_cache_result,
    _to_jsonable_scalar,
    _validate_cache_mode,
    _validate_param_grid,
    _write_cache_result,
)
from .time_splits import SeasonWalkForwardFold


def run_trainable_grid_search_three_way(
    *,
    model_factory: Callable[..., TrainablePredictiveModel],
    param_grid: Mapping[str, Sequence[Any]],
    df: pd.DataFrame,
    folds: Sequence[SeasonWalkForwardFold],
    datetime_col: str = "match_date",
    pred_home_col: str = "pred_home_goals",
    pred_away_col: str = "pred_away_goals",
    actual_home_col: str = "home_score",
    actual_away_col: str = "away_score",
    score_key: str = "avg_points",
    metric_fn: Callable[[dict[str, Any]], float] | None = None,
    cache_mode: CacheMode = "off",
    cache_dir: str = "outputs/reports/grid_search_cache",
    model_name: str | None = None,
    data_fingerprint_columns: Sequence[str] | None = None,
    show_progress: bool = True,
) -> GridSearchResult:
    """Run season-aware walk-forward grid search with train/val/eval folds.

    This variant is designed for scenarios where each fold consists of three
    disjoint parts:

    - a training slice used to fit the model,
    - a validation slice used only during fitting (for example, early
      stopping in gradient boosting),
    - an evaluation slice used exclusively for out-of-sample metrics.

    For each parameter combination and each fold:
    1. fit on train slice with ``model.fit(train_df=train_df, eval_df=val_df)``
       so that the model can use val for early stopping or calibration,
    2. predict on the evaluation slice with ``predict(eval_df)``,
    3. evaluate with ``evaluate_score_predictions``.

    Parameters
    ----------
    model_factory:
        Callable creating a trainable model from keyword parameters.
        Returned object must implement ``TrainablePredictiveModel``.
    param_grid:
        Hyperparameter grid with exactly 1 or 2 parameter names.
        Values are iterated as Cartesian product.
    df:
        Input DataFrame containing time column, model inputs and actual scores.
    folds:
        Pre-built season walk-forward folds, typically created with
        :func:`~src.models.tuning.time_splits.make_season_walk_forward_splits`.
        Each fold provides ``train_indices``, ``val_indices`` (for ``eval_df``
        in ``fit``) and ``eval_indices`` (for ``predict`` and metrics).
    datetime_col:
        Time column used for sorting. Default: ``"match_date"``.
        Column is converted with ``pd.to_datetime`` and data is sorted ascending.
    pred_home_col, pred_away_col:
        Column names expected in ``predict(eval_df)`` output.
    actual_home_col, actual_away_col:
        Ground-truth score columns in ``df`` used for evaluation.
    score_key:
        Metric key used to rank parameter sets when ``metric_fn`` is ``None``.
        Typical values:
        - ``"avg_points"`` (default),
        - ``"total_points"``,
        - ``"matches_evaluated"``,
        - ``"exact_hit_rate"``,
        - ``"goal_diff_hit_rate"``,
        - ``"outcome_hit_rate"``,
        - ``"miss_rate"``.
    metric_fn:
        Optional custom ranking function taking aggregated fold metrics dict
        and returning a numeric objective. Overrides ``score_key``.
    cache_mode:
        Cache behavior:
        - ``"off"``: no cache read/write,
        - ``"use"``: read cache if available, otherwise compute and save,
        - ``"refresh"``: force recompute and overwrite cache.
    cache_dir:
        Directory for JSON cache files.
    model_name:
        Optional stable model name used in cache key.
    data_fingerprint_columns:
        Optional list of columns used to build cache data fingerprint.
        If ``None``, defaults to time and evaluation-related columns when available.
    show_progress:
        Whether to display a text progress bar with elapsed time and ETA.

    Returns
    -------
    GridSearchResult
        Aggregated grid search results with:
        - ``results_df`` sorted by ``objective_metric`` descending,
        - ``best_params``,
        - ``best_metric``,
        - ``ranking_metric``,
        - cache metadata (hit/path).
    """
    _validate_param_grid(param_grid)
    _validate_cache_mode(cache_mode)

    sorted_df = _prepare_sorted_df_for_trainable_grid(df, datetime_col=datetime_col)

    if not folds:
        raise ValueError("folds must contain at least one SeasonWalkForwardFold.")

    ranking_metric = "custom_metric" if metric_fn is not None else score_key
    cache_path = None
    if cache_mode in {"use", "refresh"}:
        cache_model_name = _trainable_three_way_cache_model_name(
            model_name=model_name or getattr(model_factory, "__name__", "model_factory"),
            datetime_col=datetime_col,
            folds=folds,
        )
        default_fp_columns = [
            datetime_col,
            actual_home_col,
            actual_away_col,
            pred_home_col,
            pred_away_col,
        ]
        fp_columns = (
            list(data_fingerprint_columns)
            if data_fingerprint_columns is not None
            else [col for col in default_fp_columns if col in sorted_df.columns]
        )
        cache_path = _build_cache_path(
            cache_dir=cache_dir,
            model_name=cache_model_name,
            param_grid=param_grid,
            score_key=score_key,
            ranking_metric=ranking_metric,
            pred_home_col=pred_home_col,
            pred_away_col=pred_away_col,
            actual_home_col=actual_home_col,
            actual_away_col=actual_away_col,
            df=sorted_df,
            data_fingerprint_columns=fp_columns,
        )
        if cache_mode == "use" and cache_path.exists():
            return _read_cache_result(cache_path)

    combinations = _parameter_combinations(param_grid)
    total_steps = len(combinations) * len(folds)
    completed_steps = 0
    progress_start = time.perf_counter()
    if show_progress:
        _print_progress(
            description="Trainable grid search (three-way)",
            current=0,
            total=total_steps,
            started_at=progress_start,
        )

    rows: list[dict[str, Any]] = []
    for params in combinations:
        fold_metrics: list[dict[str, Any]] = []
        for fold in folds:
            model = model_factory(**params)
            if not isinstance(model, TrainablePredictiveModel):
                raise TypeError(
                    "model_factory must return an object implementing TrainablePredictiveModel."
                )

            train_df = sorted_df.iloc[fold.train_indices].copy()
            val_df = sorted_df.iloc[fold.val_indices].copy()
            eval_df = sorted_df.iloc[fold.eval_indices].copy()

            fitted_model = model.fit(train_df=train_df, eval_df=val_df)
            if not isinstance(fitted_model, PredictiveModel):
                raise TypeError("fit(...) must return an object implementing PredictiveModel.")

            pred_df = fitted_model.predict(eval_df)
            metrics = evaluate_score_predictions(
                pred_df,
                pred_home_col=pred_home_col,
                pred_away_col=pred_away_col,
                actual_home_col=actual_home_col,
                actual_away_col=actual_away_col,
            )
            fold_metrics.append(metrics)
            completed_steps += 1
            if show_progress:
                _print_progress(
                    description="Trainable grid search (three-way)",
                    current=completed_steps,
                    total=total_steps,
                    started_at=progress_start,
                )

        aggregated = _aggregate_fold_metrics(fold_metrics)
        objective = (
            float(metric_fn(aggregated))
            if metric_fn is not None
            else float(aggregated[score_key])
        )
        row = {**params, **aggregated, "objective_metric": objective, "n_folds": len(folds)}
        rows.append(row)

    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values(by="objective_metric", ascending=False).reset_index(
        drop=True
    )

    best_row = results_df.iloc[0]
    best_params = {name: _to_jsonable_scalar(best_row[name]) for name in param_grid.keys()}
    best_metric = float(best_row["objective_metric"])
    result = GridSearchResult(
        results_df=results_df,
        best_params=best_params,
        best_metric=best_metric,
        ranking_metric=ranking_metric,
        cache_hit=False,
        cache_path=str(cache_path) if cache_path is not None else None,
    )

    if cache_path is not None:
        _write_cache_result(cache_path, result)

    if show_progress:
        print()

    return result


def _aggregate_fold_metrics(fold_metrics: list[dict[str, Any]]) -> dict[str, Any]:
    if not fold_metrics:
        raise ValueError("fold_metrics cannot be empty.")

    total_matches = int(sum(int(m["matches_evaluated"]) for m in fold_metrics))
    total_points = float(sum(float(m["total_points"]) for m in fold_metrics))
    avg_points = (total_points / total_matches) if total_matches > 0 else float("nan")

    def weighted_rate(key: str) -> float:
        if total_matches <= 0:
            return float("nan")
        weighted_sum = 0.0
        for metrics in fold_metrics:
            matches = float(metrics["matches_evaluated"])
            value = float(metrics[key])
            weighted_sum += value * matches
        return weighted_sum / total_matches

    return {
        "matches_evaluated": total_matches,
        "total_points": int(round(total_points)),
        "avg_points": float(avg_points),
        "exact_hit_rate": float(weighted_rate("exact_hit_rate")),
        "goal_diff_hit_rate": float(weighted_rate("goal_diff_hit_rate")),
        "outcome_hit_rate": float(weighted_rate("outcome_hit_rate")),
        "miss_rate": float(weighted_rate("miss_rate")),
    }


def _prepare_sorted_df_for_trainable_grid(df: pd.DataFrame, *, datetime_col: str) -> pd.DataFrame:
    """Copy df, coerce datetime_col, sort ascending, reset_index."""
    if datetime_col not in df.columns:
        raise ValueError(f"Missing datetime column: '{datetime_col}'.")
    sorted_df = df.copy()
    sorted_df[datetime_col] = pd.to_datetime(sorted_df[datetime_col], errors="coerce")
    if sorted_df[datetime_col].isna().any():
        raise ValueError(f"Column '{datetime_col}' contains non-datetime values.")
    return sorted_df.sort_values(by=datetime_col).reset_index(drop=True)


def _trainable_three_way_cache_model_name(
    *,
    model_name: str,
    datetime_col: str,
    folds: Sequence[SeasonWalkForwardFold],
) -> str:
    """Build a cache key segment unique to three-way season folds."""
    n_folds = len(folds)
    min_train = int(min(f.train_indices.size for f in folds))
    min_val = int(min(f.val_indices.size for f in folds))
    min_eval = int(min(f.eval_indices.size for f in folds))
    return (
        f"{model_name}__three_way__dt_{datetime_col}__folds_{n_folds}"
        f"__train_min_{min_train}__val_min_{min_val}__eval_min_{min_eval}"
    )

