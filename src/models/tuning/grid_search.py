"""Grid search utilities for models implementing the predictive contract."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any, Callable, Literal, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
from pandas.util import hash_pandas_object
import seaborn as sns

from ..components import average_scoreline_nll
from ..evaluation import evaluate_score_predictions
from ..interfaces import PredictiveModel
from ..statistical import PoissonDixonColesModel


CacheMode = Literal["off", "use", "refresh"]


@dataclass(frozen=True)
class GridSearchResult:
    """Container with grid search outputs and metadata."""

    results_df: pd.DataFrame
    best_params: dict[str, Any]
    best_metric: float
    ranking_metric: str
    cache_hit: bool
    cache_path: str | None


def build_param_grid(
    param_specs: Mapping[str, Sequence[Any] | Mapping[str, Any]],
    *,
    deduplicate: bool = True,
) -> dict[str, list[Any]]:
    """Build validated 1D/2D parameter grid from sequences or numeric ranges.

    Supported spec formats per parameter:
    - explicit sequence, e.g. ``{"rho": [-0.1, -0.05, 0.0]}``
    - range spec with ``start`` + ``stop`` and one of:
      - ``step`` (inclusive end), e.g. ``{"start": 0.9, "stop": 1.1, "step": 0.05}``
      - ``num`` (linspace), e.g. ``{"start": 0.9, "stop": 1.1, "num": 5}``
    """
    if not param_specs:
        raise ValueError("param_specs cannot be empty.")

    grid: dict[str, list[Any]] = {}
    for name, spec in param_specs.items():
        if not name:
            raise ValueError("parameter names cannot be empty.")
        values = (
            _values_from_range_spec(spec)
            if isinstance(spec, Mapping)
            else _values_from_sequence_spec(spec)
        )
        if deduplicate:
            unique_values = list(dict.fromkeys(values))
            values = unique_values
        if len(values) == 0:
            raise ValueError(f"parameter '{name}' produced an empty values list.")
        grid[name] = [_to_jsonable_scalar(v) for v in values]

    _validate_param_grid(grid)
    return grid


def run_predictive_grid_search(
    *,
    model_factory: Callable[..., PredictiveModel],
    param_grid: Mapping[str, Sequence[Any]],
    df: pd.DataFrame,
    pred_home_col: str = "pred_home_goals",
    pred_away_col: str = "pred_away_goals",
    actual_home_col: str = "home_score",
    actual_away_col: str = "away_score",
    score_key: str = "avg_points",
    metric_fn: Callable[[dict[str, Any]], float] | None = None,
    cache_mode: CacheMode = "off",
    cache_dir: str | Path = "outputs/reports/grid_search_cache",
    model_name: str | None = None,
    data_fingerprint_columns: Sequence[str] | None = None,
    show_progress: bool = True,
) -> GridSearchResult:
    """Run grid search over 1D/2D parameter grid for predictive models.

    Parameters
    ----------
    model_factory:
        Callable creating a model from keyword parameters.
    param_grid:
        Parameter grid dictionary with exactly one or two keys.
    df:
        Input DataFrame used for prediction and evaluation.
    pred_home_col, pred_away_col:
        Prediction output columns used for score evaluation.
    actual_home_col, actual_away_col:
        Ground-truth score columns in the same DataFrame.
    score_key:
        Metric key from ``evaluate_score_predictions`` used for ranking when
        ``metric_fn`` is not provided.
        Typical values:
        - ``"avg_points"`` (default),
        - ``"total_points"``,
        - ``"matches_evaluated"``,
        - ``"exact_hit_rate"``,
        - ``"goal_diff_hit_rate"``,
        - ``"outcome_hit_rate"``,
        - ``"miss_rate"``.
        You may also pass any numeric key returned by
        ``evaluate_score_predictions``.
    metric_fn:
        Optional custom ranking function accepting metric dictionary.
        When provided, it takes precedence over ``score_key``.
    cache_mode:
        Cache behavior:
        - ``"off"``: no cache read/write,
        - ``"use"``: read cache if exists, else compute and save,
        - ``"refresh"``: always recompute and overwrite cache.
    cache_dir:
        Directory with cached JSON results.
    model_name:
        Optional stable model name for cache key. If ``None``, falls back to
        the factory callable name.
    data_fingerprint_columns:
        Columns used to build input fingerprint. If ``None``, all columns
        are used.
    show_progress:
        Whether to display a text progress bar with elapsed time and ETA.
    """
    _validate_param_grid(param_grid)
    _validate_cache_mode(cache_mode)

    ranking_metric = "custom_metric" if metric_fn is not None else score_key
    cache_path: Path | None = None
    if cache_mode in {"use", "refresh"}:
        cache_path = _build_cache_path(
            cache_dir=cache_dir,
            model_name=model_name or getattr(model_factory, "__name__", "model_factory"),
            param_grid=param_grid,
            score_key=score_key,
            ranking_metric=ranking_metric,
            pred_home_col=pred_home_col,
            pred_away_col=pred_away_col,
            actual_home_col=actual_home_col,
            actual_away_col=actual_away_col,
            df=df,
            data_fingerprint_columns=data_fingerprint_columns,
        )
        if cache_mode == "use" and cache_path.exists():
            return _read_cache_result(cache_path)

    combinations = _parameter_combinations(param_grid)
    result_rows: list[dict[str, Any]] = []
    progress_start = time.perf_counter()
    if show_progress:
        _print_progress(
            description="Predictive grid search",
            current=0,
            total=len(combinations),
            started_at=progress_start,
        )

    for index, params in enumerate(combinations, start=1):
        model = model_factory(**params)
        if not isinstance(model, PredictiveModel):
            raise TypeError("model_factory must return an object implementing PredictiveModel.")

        pred_df = model.predict(df)
        metrics = evaluate_score_predictions(
            pred_df,
            pred_home_col=pred_home_col,
            pred_away_col=pred_away_col,
            actual_home_col=actual_home_col,
            actual_away_col=actual_away_col,
        )
        objective = (
            float(metric_fn(metrics)) if metric_fn is not None else float(metrics[score_key])
        )

        row = {**params, **metrics, "objective_metric": objective}
        result_rows.append(row)
        if show_progress:
            _print_progress(
                description="Predictive grid search",
                current=index,
                total=len(combinations),
                started_at=progress_start,
            )
    if show_progress:
        print()

    results_df = pd.DataFrame(result_rows)
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

    return result


def run_predictive_nll_grid_search(
    *,
    model_factory: Callable[..., PredictiveModel],
    param_grid: Mapping[str, Sequence[Any]],
    df: pd.DataFrame,
    actual_home_col: str = "home_score",
    actual_away_col: str = "away_score",
    exp_goals_home_col: str = "exp_goals_home",
    exp_goals_away_col: str = "exp_goals_away",
    cache_mode: CacheMode = "off",
    cache_dir: str | Path = "outputs/reports/grid_search_cache",
    model_name: str | None = None,
    data_fingerprint_columns: Sequence[str] | None = None,
    show_progress: bool = True,
) -> GridSearchResult:
    """Grid search for :class:`~src.models.statistical.PoissonDixonColesModel` ranking on NLL.

    For each parameter combination, runs ``model.predict`` and measures the
    mean negative log-likelihood of the true scoreline under
    :class:`~src.models.components.PoissonMatrixBuilder` with the model's
    ``rho`` and ``max_goals_matrix``, using expected goals from
    ``exp_goals_home_col`` and ``exp_goals_away_col`` in the prediction frame.

    Lower ``objective_metric`` (equal to average NLL) is better. The returned
    ``GridSearchResult`` is sorted with the best (lowest NLL) row first. Use
    :func:`plot_grid_search_2d` with ``metric_name="objective_metric"`` to
    visualize 2D grids.

    Parameters
    ----------
    model_factory:
        Must return a :class:`PoissonDixonColesModel` instance.
    param_grid, df:
        Same as :func:`run_predictive_grid_search` (1 or 2 tunable parameters).
    actual_home_col, actual_away_col:
        Observed goal columns in ``df`` (also present in ``model.predict`` output).
    exp_goals_home_col, exp_goals_away_col:
        Column names in the prediction output with per-row expected goals.
    cache_mode, cache_dir, model_name, data_fingerprint_columns, show_progress:
        Same meaning as in :func:`run_predictive_grid_search`. Cache key
        includes ``objective: scoreline_nll`` so it does not collide with
        point-based grid search.
    """
    _validate_param_grid(param_grid)
    _validate_cache_mode(cache_mode)

    ranking_metric = "avg_nll"
    score_key = "avg_nll"
    cache_path: Path | None = None
    if cache_mode in {"use", "refresh"}:
        cache_path = _build_cache_path(
            cache_dir=cache_dir,
            model_name=model_name or getattr(model_factory, "__name__", "model_factory"),
            param_grid=param_grid,
            score_key=score_key,
            ranking_metric=ranking_metric,
            pred_home_col=exp_goals_home_col,
            pred_away_col=exp_goals_away_col,
            actual_home_col=actual_home_col,
            actual_away_col=actual_away_col,
            df=df,
            data_fingerprint_columns=data_fingerprint_columns,
            cache_payload_extras={"objective": "scoreline_nll"},
        )
        if cache_mode == "use" and cache_path.exists():
            return _read_cache_result(cache_path)

    combinations = _parameter_combinations(param_grid)
    result_rows: list[dict[str, Any]] = []
    progress_start = time.perf_counter()
    if show_progress:
        _print_progress(
            description="Predictive NLL grid search",
            current=0,
            total=len(combinations),
            started_at=progress_start,
        )

    for index, params in enumerate(combinations, start=1):
        model = model_factory(**params)
        if not isinstance(model, PoissonDixonColesModel):
            raise TypeError(
                "run_predictive_nll_grid_search requires model_factory to return "
                "a PoissonDixonColesModel instance.",
            )
        if not isinstance(model, PredictiveModel):
            raise TypeError("model_factory must return an object implementing PredictiveModel.")

        pred_df = model.predict(df)
        for col in (exp_goals_home_col, exp_goals_away_col, actual_home_col, actual_away_col):
            if col not in pred_df.columns:
                raise ValueError(f"Column '{col}' not found in model prediction output.")

        eval_data = pred_df[[exp_goals_home_col, exp_goals_away_col, actual_home_col, actual_away_col]].copy()
        eval_data = eval_data.replace([np.inf, -np.inf], np.nan)
        eval_data = eval_data.dropna(
            subset=[exp_goals_home_col, exp_goals_away_col, actual_home_col, actual_away_col]
        )
        if eval_data.empty:
            raise ValueError("No valid rows for NLL after dropping NaN in goals columns.")

        lam_h = eval_data[exp_goals_home_col].to_numpy(dtype=np.float64)
        lam_a = eval_data[exp_goals_away_col].to_numpy(dtype=np.float64)
        act_h = eval_data[actual_home_col].to_numpy(dtype=np.float64)
        act_a = eval_data[actual_away_col].to_numpy(dtype=np.float64)
        act_h = np.rint(act_h).astype(np.intp)
        act_a = np.rint(act_a).astype(np.intp)

        mean_nll, n_used = average_scoreline_nll(
            lam_h, lam_a, act_h, act_a,
            rho=float(model.rho), max_goals_matrix=int(model.max_goals_matrix),
        )

        row = {
            **params,
            "objective_metric": float(mean_nll),
            "avg_nll": float(mean_nll),
            "matches_nll": int(n_used),
        }
        result_rows.append(row)
        if show_progress:
            _print_progress(
                description="Predictive NLL grid search",
                current=index,
                total=len(combinations),
                started_at=progress_start,
            )
    if show_progress:
        print()

    results_df = pd.DataFrame(result_rows)
    results_df = results_df.sort_values(by="objective_metric", ascending=True).reset_index(
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

    return result


def plot_grid_search_1d(
    results_df: pd.DataFrame,
    *,
    param_name: str,
    metric_name: str = "objective_metric",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot 1D grid search results as line chart with best point marker."""
    _validate_plot_columns(results_df, [param_name, metric_name])
    plot_df = results_df[[param_name, metric_name]].copy().sort_values(param_name)
    metric_decimals = _infer_metric_decimals(plot_df[metric_name], metric_name=metric_name)

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    ax.plot(plot_df[param_name], plot_df[metric_name], marker="o")
    best_idx = int(plot_df[metric_name].idxmax())
    best_x = plot_df.loc[best_idx, param_name]
    best_y = plot_df.loc[best_idx, metric_name]
    best_y_label = _format_number(best_y, decimals=metric_decimals)
    ax.scatter([best_x], [best_y], s=80, marker="*", label=f"best={best_y_label}")
    ax.set_title(f"Grid Search 1D: {param_name}")
    ax.set_xlabel(param_name)
    ax.set_ylabel(metric_name)
    if pd.api.types.is_numeric_dtype(plot_df[param_name]):
        x_values = plot_df[param_name].tolist()
        ax.set_xticks(x_values)
        ax.set_xticklabels([_format_param_value(x) for x in x_values])
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda value, _: _format_number(value, decimals=metric_decimals))
    )
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def _heatmap_cmap(cmap: str, *, reverse: bool) -> str:
    """Return a matplotlib colormap name, optionally reversed (``*_r``)."""
    if not reverse:
        return cmap
    if cmap.endswith("_r"):
        return cmap
    return f"{cmap}_r"


def plot_grid_search_2d(
    results_df: pd.DataFrame,
    *,
    x_param: str,
    y_param: str,
    metric_name: str = "objective_metric",
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    reverse_colormap: bool = False,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot 2D grid search results as heatmap.

    Parameters
    ----------
    vmin, vmax:
        Optional limits for the colormap (forwarded to ``seaborn.heatmap``).
        Use them when a few outlying values compress contrast—e.g. for NLL
        heatmaps, set ``vmax`` near the bulk of the grid so local minima show up.
    cmap:
        Colormap name accepted by ``seaborn.heatmap`` / matplotlib.
    reverse_colormap:
        When ``True``, uses the reversed colormap (e.g. ``viridis`` →
        ``viridis_r``) so that **low** metric values use the end of the scale
        that is visually lighter. Useful for metrics where lower is better (NLL);
        prefer :func:`plot_nll_grid_search_2d` for NLL with sensible defaults.
    """
    _validate_plot_columns(results_df, [x_param, y_param, metric_name])
    pivot_df = results_df.pivot(index=y_param, columns=x_param, values=metric_name)
    pivot_df = pivot_df.sort_index(axis=0).sort_index(axis=1)
    metric_decimals = _infer_metric_decimals(pivot_df.to_numpy().reshape(-1), metric_name=metric_name)
    annot_df = pivot_df.apply(
        lambda column: column.map(lambda value: _format_number(value, decimals=metric_decimals))
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(9, 6))

    heatmap_cmap = _heatmap_cmap(cmap, reverse=reverse_colormap)
    heatmap_kwargs: dict[str, Any] = {
        "annot": annot_df,
        "fmt": "",
        "cmap": heatmap_cmap,
        "ax": ax,
    }
    if vmin is not None:
        heatmap_kwargs["vmin"] = float(vmin)
    if vmax is not None:
        heatmap_kwargs["vmax"] = float(vmax)
    sns.heatmap(pivot_df, **heatmap_kwargs)
    ax.set_title(f"Grid Search 2D: {y_param} vs {x_param}")
    ax.set_xlabel(x_param)
    ax.set_ylabel(y_param)
    ax.set_xticklabels([_format_param_value(value) for value in pivot_df.columns], rotation=45, ha="right")
    ax.set_yticklabels([_format_param_value(value) for value in pivot_df.index], rotation=0)
    return ax


def plot_nll_grid_search_2d(
    results_df: pd.DataFrame,
    *,
    x_param: str,
    y_param: str,
    metric_name: str = "objective_metric",
    high_quantile: float | None = 0.95,
    low_quantile: float | None = None,
    reverse_colormap: bool = True,
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot 2D grid search from :func:`run_predictive_nll_grid_search` with NLL-friendly defaults.

    Sets color limits from the metric column so outliers do not flatten contrast:
    ``vmin`` from the column minimum (or a lower quantile), ``vmax`` from
    ``high_quantile`` (default 0.95) when not ``None``. By default uses a
    **reversed** colormap so **lower** NLL (better) maps to **lighter** colors.

    Internally calls :func:`plot_grid_search_2d`.

    Parameters
    ----------
    high_quantile:
        Upper quantile for ``vmax``. Use ``None`` to leave ``vmax`` unset
        (full data range for the upper end).
    low_quantile:
        If ``None`` (default), ``vmin`` is the column minimum. Otherwise
        ``vmin`` is ``metric.quantile(low_quantile)`` (e.g. ``0.05`` for a
        robust lower bound).
    reverse_colormap:
        Default ``True``: low NLL is drawn with the lighter end of the scale.
    """
    _validate_plot_columns(results_df, [x_param, y_param, metric_name])
    series = results_df[metric_name]
    if low_quantile is not None:
        lq = float(low_quantile)
        if not 0.0 <= lq <= 1.0:
            raise ValueError("low_quantile must be in [0, 1].")
    if low_quantile is None:
        vmin: float | None = float(series.min())
    else:
        vmin = float(series.quantile(float(low_quantile)))
    if high_quantile is None:
        vmax: float | None = None
    else:
        hq = float(high_quantile)
        if not 0.0 < hq <= 1.0:
            raise ValueError("high_quantile must be in (0, 1] when not None.")
        vmax = float(series.quantile(hq))
    if vmin is not None and vmax is not None and vmax <= vmin:
        vmin, vmax = None, None
    return plot_grid_search_2d(
        results_df,
        x_param=x_param,
        y_param=y_param,
        metric_name=metric_name,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        reverse_colormap=reverse_colormap,
        ax=ax,
    )


def _validate_param_grid(param_grid: Mapping[str, Sequence[Any]]) -> None:
    if not param_grid:
        raise ValueError("param_grid cannot be empty.")
    if len(param_grid) not in {1, 2}:
        raise ValueError("param_grid must contain exactly 1 or 2 parameters.")
    for name, values in param_grid.items():
        if not name:
            raise ValueError("parameter names cannot be empty.")
        if len(values) == 0:
            raise ValueError(f"parameter '{name}' must define at least one value.")


def _validate_cache_mode(cache_mode: CacheMode) -> None:
    if cache_mode not in {"off", "use", "refresh"}:
        raise ValueError("cache_mode must be one of: 'off', 'use', 'refresh'.")


def _parameter_combinations(param_grid: Mapping[str, Sequence[Any]]) -> list[dict[str, Any]]:
    names = list(param_grid.keys())
    values_product = itertools.product(*(param_grid[name] for name in names))
    return [dict(zip(names, values, strict=True)) for values in values_product]


def _build_cache_path(
    *,
    cache_dir: str | Path,
    model_name: str,
    param_grid: Mapping[str, Sequence[Any]],
    score_key: str,
    ranking_metric: str,
    pred_home_col: str,
    pred_away_col: str,
    actual_home_col: str,
    actual_away_col: str,
    df: pd.DataFrame,
    data_fingerprint_columns: Sequence[str] | None,
    cache_payload_extras: Mapping[str, Any] | None = None,
) -> Path:
    safe_model_name = _safe_filename_part(model_name)
    fp_columns = list(df.columns if data_fingerprint_columns is None else data_fingerprint_columns)
    missing = [col for col in fp_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing fingerprint columns in DataFrame: {missing}")

    payload: dict[str, Any] = {
        "model_name": model_name,
        "param_grid": {k: list(v) for k, v in param_grid.items()},
        "score_key": score_key,
        "ranking_metric": ranking_metric,
        "pred_home_col": pred_home_col,
        "pred_away_col": pred_away_col,
        "actual_home_col": actual_home_col,
        "actual_away_col": actual_away_col,
        "data_fingerprint_columns": fp_columns,
        "data_fingerprint": _fingerprint_dataframe(df, fp_columns),
    }
    if cache_payload_extras is not None:
        for extra_key, extra_val in cache_payload_extras.items():
            payload[extra_key] = extra_val
    payload_json = json.dumps(payload, sort_keys=True, default=str)
    key = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()[:16]

    cache_dir_path = Path(cache_dir)
    return cache_dir_path / f"grid_search_{safe_model_name}_{key}.json"


def _fingerprint_dataframe(df: pd.DataFrame, columns: Sequence[str]) -> str:
    data = df.loc[:, list(columns)].copy()
    object_cols = data.select_dtypes(include=["object"]).columns
    for col in object_cols:
        data[col] = data[col].astype(str)
    digest_source = hash_pandas_object(data, index=True).values.tobytes()
    return hashlib.sha256(digest_source).hexdigest()


def _write_cache_result(cache_path: Path, result: GridSearchResult) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "best_params": result.best_params,
        "best_metric": result.best_metric,
        "ranking_metric": result.ranking_metric,
        "results_records": _records_to_jsonable(result.results_df.to_dict(orient="records")),
        "cache_path": str(cache_path),
    }
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _read_cache_result(cache_path: Path) -> GridSearchResult:
    with cache_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    results_df = pd.DataFrame(payload["results_records"])
    return GridSearchResult(
        results_df=results_df,
        best_params=dict(payload["best_params"]),
        best_metric=float(payload["best_metric"]),
        ranking_metric=str(payload["ranking_metric"]),
        cache_hit=True,
        cache_path=str(cache_path),
    )


def _records_to_jsonable(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    jsonable: list[dict[str, Any]] = []
    for record in records:
        converted: dict[str, Any] = {}
        for key, value in record.items():
            converted[key] = _to_jsonable_scalar(value)
        jsonable.append(converted)
    return jsonable


def _to_jsonable_scalar(value: Any) -> Any:
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


def _validate_plot_columns(results_df: pd.DataFrame, required_columns: Sequence[str]) -> None:
    missing = [col for col in required_columns if col not in results_df.columns]
    if missing:
        raise ValueError(f"Missing columns in results_df: {missing}")


def _safe_filename_part(value: str) -> str:
    sanitized = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)
    return sanitized.strip("_") or "model"


def _values_from_sequence_spec(spec: Sequence[Any]) -> list[Any]:
    values = list(spec)
    if len(values) == 0:
        raise ValueError("sequence-based parameter spec cannot be empty.")
    _validate_numeric_values(values)
    return values


def _values_from_range_spec(spec: Mapping[str, Any]) -> list[Any]:
    required = {"start", "stop"}
    missing = [key for key in required if key not in spec]
    if missing:
        raise ValueError(f"range spec is missing required keys: {missing}")

    has_step = "step" in spec
    has_num = "num" in spec
    if has_step == has_num:
        raise ValueError("range spec must define exactly one of: 'step' or 'num'.")

    start = float(spec["start"])
    stop = float(spec["stop"])

    if has_step:
        step = float(spec["step"])
        if step == 0:
            raise ValueError("range spec 'step' cannot be 0.")
        if (stop - start) * step < 0:
            raise ValueError("range spec 'step' sign does not move from start to stop.")
        epsilon = abs(step) * 1e-9
        values = np.arange(start, stop + epsilon, step, dtype=float).tolist()
    else:
        num = int(spec["num"])
        if num <= 0:
            raise ValueError("range spec 'num' must be greater than 0.")
        values = np.linspace(start, stop, num=num, dtype=float).tolist()

    _validate_numeric_values(values)
    return values


def _validate_numeric_values(values: Sequence[Any]) -> None:
    for value in values:
        if isinstance(value, (int, float, np.integer, np.floating)):
            if not np.isfinite(float(value)):
                raise ValueError("parameter values must be finite numbers.")


def _infer_metric_decimals(values: Sequence[Any], *, metric_name: str) -> int:
    if metric_name in {"total_points", "matches_evaluated"}:
        return 0
    numeric_values = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna()
    if len(numeric_values) == 0:
        return 3
    is_integer_like = np.all(np.isclose(numeric_values.to_numpy(), np.round(numeric_values.to_numpy())))
    return 0 if is_integer_like else 3


def _format_number(value: Any, *, decimals: int) -> str:
    if pd.isna(value):
        return "NaN"
    number = float(value)
    if decimals <= 0:
        return str(int(round(number)))
    text = f"{number:.{decimals}f}"
    return text.rstrip("0").rstrip(".")


def _format_param_value(value: Any) -> str:
    if pd.isna(value):
        return "NaN"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        text = np.format_float_positional(float(value), precision=10, trim="-")
        return "0" if text in {"-0", "-0.0"} else text
    return str(value)


def _print_progress(
    *,
    description: str,
    current: int,
    total: int,
    started_at: float,
    width: int = 30,
) -> None:
    safe_total = max(total, 1)
    bounded_current = min(max(current, 0), safe_total)
    ratio = bounded_current / safe_total
    filled = int(width * ratio)
    bar = f"{'=' * filled}{'-' * (width - filled)}"
    elapsed = max(time.perf_counter() - started_at, 0.0)
    speed = (bounded_current / elapsed) if elapsed > 0 else 0.0
    remaining = ((safe_total - bounded_current) / speed) if speed > 0 else None
    eta_text = _format_seconds(remaining)
    elapsed_text = _format_seconds(elapsed)
    print(
        f"\r{description}: [{bar}] {bounded_current}/{safe_total} "
        f"| elapsed {elapsed_text} | eta {eta_text}",
        end="",
        flush=True,
    )


def _format_seconds(seconds: float | None) -> str:
    if seconds is None:
        return "--:--"
    whole_seconds = max(int(round(seconds)), 0)
    hours, remainder = divmod(whole_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"
