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
