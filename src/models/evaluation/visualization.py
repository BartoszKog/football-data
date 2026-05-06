"""Visualization helpers for scoreline prediction evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .scoring import ScoreRule, compute_points_per_match


_OUTCOME_ORDER = ("1", "X", "2")


def _outcome_1x2(home_goals: int, away_goals: int) -> str:
    """Map scoreline to 1x2 outcome label."""
    if home_goals > away_goals:
        return "1"
    if home_goals == away_goals:
        return "X"
    return "2"


@dataclass
class PointsSummary1x2:
    """Summary of prediction evaluation: points and 1x2 outcome matrix."""

    total_points: float
    mean_points: float
    points_distribution: pd.Series
    outcome_matrix: pd.DataFrame


def summarize_predictions_1x2(
    df: pd.DataFrame,
    *,
    pred_home_col: str = "pred_home_goals",
    pred_away_col: str = "pred_away_goals",
    actual_home_col: str = "home_score",
    actual_away_col: str = "away_score",
    rule: ScoreRule | None = None,
) -> PointsSummary1x2:
    """Compute points and 1x2 outcome matrix from scoreline predictions.

    Parameters
    ----------
    df:
        DataFrame with prediction and actual result columns.
    pred_home_col, pred_away_col:
        Column names with model score predictions.
    actual_home_col, actual_away_col:
        Column names with observed match scores.
    rule:
        Optional custom scoring rule.

    Returns
    -------
    PointsSummary1x2
        Aggregated metrics: total_points, mean_points, points_distribution,
        outcome_matrix (crosstab actual vs predicted 1x2).
    """
    points = compute_points_per_match(
        df,
        pred_home_col=pred_home_col,
        pred_away_col=pred_away_col,
        actual_home_col=actual_home_col,
        actual_away_col=actual_away_col,
        rule=rule,
    )

    if points.empty:
        return PointsSummary1x2(
            total_points=0.0,
            mean_points=float("nan"),
            points_distribution=pd.Series(dtype=int),
            outcome_matrix=pd.DataFrame(),
        )

    total_points = float(points.sum())
    mean_points = float(points.mean())
    points_distribution = points.value_counts().sort_index()

    data = df.loc[points.index].copy()
    pred_home = np.rint(data[pred_home_col].to_numpy(dtype=float)).astype(int)
    pred_away = np.rint(data[pred_away_col].to_numpy(dtype=float)).astype(int)
    actual_home = np.rint(data[actual_home_col].to_numpy(dtype=float)).astype(int)
    actual_away = np.rint(data[actual_away_col].to_numpy(dtype=float)).astype(int)

    actual_1x2 = pd.Series(
        [_outcome_1x2(int(h), int(a)) for h, a in zip(actual_home, actual_away)],
        index=points.index,
    )
    pred_1x2 = pd.Series(
        [_outcome_1x2(int(h), int(a)) for h, a in zip(pred_home, pred_away)],
        index=points.index,
    )

    outcome_matrix = pd.crosstab(actual_1x2, pred_1x2)
    outcome_matrix = outcome_matrix.reindex(index=_OUTCOME_ORDER, columns=_OUTCOME_ORDER).fillna(0).astype(int)

    return PointsSummary1x2(
        total_points=total_points,
        mean_points=mean_points,
        points_distribution=points_distribution,
        outcome_matrix=outcome_matrix,
    )


def plot_predictions_summary(
    df: pd.DataFrame,
    *,
    pred_home_col: str = "pred_home_goals",
    pred_away_col: str = "pred_away_goals",
    actual_home_col: str = "home_score",
    actual_away_col: str = "away_score",
    rule: ScoreRule | None = None,
    model_name: str | None = None,
    figsize: tuple[float, float] = (14, 5),
) -> plt.Figure:
    """Plot points distribution and 1x2 outcome matrix for scoreline predictions.

    Parameters
    ----------
    df:
        DataFrame with prediction and actual result columns.
    pred_home_col, pred_away_col:
        Column names with model score predictions.
    actual_home_col, actual_away_col:
        Column names with observed match scores.
    rule:
        Optional custom scoring rule.
    model_name:
        Optional label for the plot title.
    figsize:
        Figure size (width, height).

    Returns
    -------
    matplotlib.figure.Figure
        The figure containing the bar chart and heatmap.
    """
    summary = summarize_predictions_1x2(
        df,
        pred_home_col=pred_home_col,
        pred_away_col=pred_away_col,
        actual_home_col=actual_home_col,
        actual_away_col=actual_away_col,
        rule=rule,
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # Bar chart: points distribution
    if summary.points_distribution.empty:
        ax1.text(0.5, 0.5, "Brak danych", ha="center", va="center", transform=ax1.transAxes)
    else:
        dist = summary.points_distribution.sort_index()
        # Kolory: dla klasycznego schematu 0–3 użyj stałej palety,
        # w innym przypadku zastosuj ciągłą paletę kolorów.
        if set(dist.index.tolist()) <= {0, 1, 2, 3}:
            color_map = {0: "red", 1: "orange", 2: "lightgreen", 3: "green"}
            colors = [color_map.get(int(x), "grey") for x in dist.index]
        else:
            colors = sns.color_palette("viridis", n_colors=len(dist))

        bars = ax1.bar(dist.index, dist.values, color=colors)
        ax1.bar_label(bars)
        ax1.set_xticks(dist.index)
        ax1.set_xlabel("Punkty")
        ax1.set_ylabel("Liczba meczów")

    title = "Rozkład zdobywanych punktów (0, 1, 2, 3)"
    if model_name:
        title += f" – {model_name}"
    ax1.set_title(title)

    # Summary text box – pokazuj tylko, gdy mamy dane i skończoną średnią
    if not summary.points_distribution.empty and np.isfinite(summary.mean_points):
        text = f"Suma punktów: {summary.total_points:.0f}\nŚrednio na mecz: {summary.mean_points:.3f}"
        ax1.text(
            0.95,
            0.95,
            text,
            transform=ax1.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    # Heatmap: 1x2 outcome matrix
    if summary.outcome_matrix.empty:
        ax2.text(0.5, 0.5, "Brak danych", ha="center", va="center", transform=ax2.transAxes)
    else:
        sns.heatmap(
            summary.outcome_matrix,
            annot=True,
            fmt="d",
            cmap="Blues",
            ax=ax2,
        )
    ax2.set_xlabel("Typ modelu")
    ax2.set_ylabel("Rzeczywisty wynik")
    ax2.set_title("Macierz pomyłek 1X2")

    plt.tight_layout()
    return fig


def plot_predictions_scoreline_summary(
    df: pd.DataFrame,
    *,
    pred_home_col: str = "pred_home_goals",
    pred_away_col: str = "pred_away_goals",
    actual_home_col: str = "home_score",
    actual_away_col: str = "away_score",
    max_goals_clip: int = 4,
    top_n_scorelines: int = 6,
    model_name: str | None = None,
    figsize: tuple[float, float] = (14, 9),
) -> plt.Figure:
    """Plot predicted vs actual scorelines: goal-pair heatmaps and top-N score counts.

    Uses rows where all four columns are finite after coercion to numeric.

    The upper row shows ``(home, away)`` goal-count grids with values clipped at
    ``max_goals_clip`` (scores ``>= max_goals_clip`` are mapped to that bucket).
    Axis ticks use ``0 … max_goals_clip - 1`` and ``+max_goals_clip`` for the top bucket.
    Both heatmaps share the same color scale upper bound for comparability.

    The lower row shows horizontal bar charts of the ``top_n_scorelines`` most
    frequent ``h:a`` strings from rounded predictions and from actual scores
    (no clipping — full integer scorelines).

    Parameters
    ----------
    df:
        DataFrame with prediction and observed score columns.
    pred_home_col, pred_away_col:
        Predicted goals columns (rounded with ``rint`` for heatmaps and bars).
    actual_home_col, actual_away_col:
        Observed goals columns (converted to integers for heatmaps and bars).
    max_goals_clip:
        Heatmap axis runs ``0 .. max_goals_clip``; higher totals are bucketed.
        Tick labels show ``+max_goals_clip`` on the last bin (not plain digits).
    top_n_scorelines:
        How many distinct ``h:a`` labels to show per bottom panel.
    model_name:
        Optional figure title (``suptitle``).
    figsize:
        Passed to ``plt.subplots(2, 2, ...)``. Default height is larger than
        ``plot_predictions_summary`` because this figure has four axes.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with axes ``[heatmap pred, heatmap actual; bars pred, bars actual]``.
    """
    cols = [pred_home_col, pred_away_col, actual_home_col, actual_away_col]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns for scoreline summary plot: {missing}")

    data = df.loc[:, cols].copy()
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors="coerce")
    data = data.dropna(how="any")
    if data.empty:
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        for ax in axes.flat:
            ax.text(0.5, 0.5, "Brak danych", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        if model_name:
            fig.suptitle(model_name)
        plt.tight_layout()
        return fig

    pred_h = np.rint(data[pred_home_col].to_numpy(dtype=float)).astype(int)
    pred_a = np.rint(data[pred_away_col].to_numpy(dtype=float)).astype(int)
    act_h = np.rint(data[actual_home_col].to_numpy(dtype=float)).astype(int)
    act_a = np.rint(data[actual_away_col].to_numpy(dtype=float)).astype(int)

    ph = np.minimum(pred_h, max_goals_clip)
    pa = np.minimum(pred_a, max_goals_clip)
    ah = np.minimum(act_h, max_goals_clip)
    aa = np.minimum(act_a, max_goals_clip)

    levels = list(range(max_goals_clip + 1))
    heatmap_tick_labels = [
        str(k) if k < max_goals_clip else f"+{max_goals_clip}"
        for k in levels
    ]
    full_idx = pd.MultiIndex.from_product([levels, levels], names=["h", "a"])

    def _heatmap_counts(h_arr: np.ndarray, a_arr: np.ndarray) -> pd.DataFrame:
        stacked = (
            pd.DataFrame({"h": h_arr, "a": a_arr})
            .groupby(["h", "a"], observed=False)
            .size()
            .reindex(full_idx, fill_value=0)
            .rename("n")
            .reset_index()
        )
        wide = stacked.pivot(index="a", columns="h", values="n").reindex(
            index=levels, columns=levels
        )
        return wide.fillna(0.0)

    wide_pred = _heatmap_counts(ph, pa)
    wide_act = _heatmap_counts(ah, aa)
    pred_mat = wide_pred.to_numpy(dtype=float)
    act_mat = wide_act.to_numpy(dtype=float)
    vmax = float(max(pred_mat.max(), act_mat.max(), 1.0))

    pred_lines = pd.Series([f"{int(h)}:{int(a)}" for h, a in zip(pred_h, pred_a)]).value_counts()
    act_lines = pd.Series([f"{int(h)}:{int(a)}" for h, a in zip(act_h, act_a)]).value_counts()
    top_pred = pred_lines.head(top_n_scorelines)
    top_act = act_lines.head(top_n_scorelines)

    fig, axes = plt.subplots(2, 2, figsize=figsize)
    ax00, ax01 = axes[0]
    ax10, ax11 = axes[1]

    sns.heatmap(
        wide_pred,
        ax=ax00,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        vmin=0.0,
        vmax=vmax,
        cbar_kws={"label": "Liczba meczów"},
        linewidths=0.5,
        linecolor="white",
    )
    ax00.set_xticklabels(heatmap_tick_labels, rotation=0)
    ax00.set_yticklabels(heatmap_tick_labels, rotation=0)
    ax00.set_xlabel("Bramki gospodarza (predykcja)")
    ax00.set_ylabel("Bramki gościa (predykcja)")
    ax00.set_title("Typowane pary bramek")

    sns.heatmap(
        wide_act,
        ax=ax01,
        annot=True,
        fmt=".0f",
        cmap="Blues",
        vmin=0.0,
        vmax=vmax,
        cbar_kws={"label": "Liczba meczów"},
        linewidths=0.5,
        linecolor="white",
    )
    ax01.set_xticklabels(heatmap_tick_labels, rotation=0)
    ax01.set_yticklabels(heatmap_tick_labels, rotation=0)
    ax01.set_xlabel("Bramki gospodarza (rzeczywistość)")
    ax01.set_ylabel("Bramki gościa (rzeczywistość)")
    ax01.set_title("Rzeczywiste pary bramek")

    if top_pred.empty:
        ax10.text(0.5, 0.5, "Brak danych", ha="center", va="center", transform=ax10.transAxes)
        ax10.set_axis_off()
    else:
        y_pos = np.arange(len(top_pred))
        bars = ax10.barh(y_pos, top_pred.to_numpy(dtype=float), color="#4e79a7")
        ax10.bar_label(bars, fmt="%.0f", padding=3)
        ax10.set_yticks(y_pos)
        ax10.set_yticklabels(top_pred.index.tolist())
        ax10.invert_yaxis()
        ax10.set_xlabel("Liczba meczów")
        ax10.set_title(f"Top {top_n_scorelines} typowanych wyników (h:a)")
        ax10.set_ylabel(None)

    if top_act.empty:
        ax11.text(0.5, 0.5, "Brak danych", ha="center", va="center", transform=ax11.transAxes)
        ax11.set_axis_off()
    else:
        y_pos = np.arange(len(top_act))
        bars = ax11.barh(y_pos, top_act.to_numpy(dtype=float), color="#f28e2b")
        ax11.bar_label(bars, fmt="%.0f", padding=3)
        ax11.set_yticks(y_pos)
        ax11.set_yticklabels(top_act.index.tolist())
        ax11.invert_yaxis()
        ax11.set_xlabel("Liczba meczów")
        ax11.set_title(f"Top {top_n_scorelines} rzeczywistych wyników (h:a)")
        ax11.set_ylabel(None)

    if model_name:
        fig.suptitle(model_name)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.96 if model_name else 1.0])
    return fig
