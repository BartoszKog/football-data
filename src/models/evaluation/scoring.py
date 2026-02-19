"""Reusable scoreline evaluation helpers for prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreRule:
    """Point schema for scoreline evaluation."""

    exact: int = 3
    goal_diff: int = 2
    outcome: int = 1
    miss: int = 0


def score_single_prediction(
    pred_home: int,
    pred_away: int,
    actual_home: int,
    actual_away: int,
    *,
    rule: ScoreRule | None = None,
) -> int:
    """Return points for one predicted scoreline.

    The function applies mutually exclusive tiers:
    1. exact score,
    2. correct goal difference,
    3. correct 1x2 outcome,
    4. miss.
    """
    scoring_rule = rule or ScoreRule()

    if pred_home == actual_home and pred_away == actual_away:
        return scoring_rule.exact

    pred_diff = pred_home - pred_away
    actual_diff = actual_home - actual_away
    if pred_diff == actual_diff:
        return scoring_rule.goal_diff

    pred_outcome = int(np.sign(pred_diff))
    actual_outcome = int(np.sign(actual_diff))
    if pred_outcome == actual_outcome:
        return scoring_rule.outcome

    return scoring_rule.miss


def evaluate_score_predictions(
    df: pd.DataFrame,
    *,
    pred_home_col: str = "pred_home_goals",
    pred_away_col: str = "pred_away_goals",
    actual_home_col: str = "home_score",
    actual_away_col: str = "away_score",
    rule: ScoreRule | None = None,
) -> dict[str, Any]:
    """Evaluate scoreline predictions and return aggregate metrics.

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
    dict[str, Any]
        Metrics dictionary with:
        - ``matches_evaluated``
        - ``total_points``
        - ``avg_points``
        - ``exact_hit_rate``
        - ``goal_diff_hit_rate``
        - ``outcome_hit_rate``
        - ``miss_rate``
    """
    scoring_rule = rule or ScoreRule()

    required_cols = [pred_home_col, pred_away_col, actual_home_col, actual_away_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns for evaluation: {missing_cols}")

    data = df.copy()
    for column in required_cols:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data.dropna(subset=required_cols)
    if data.empty:
        return {
            "matches_evaluated": 0,
            "total_points": 0,
            "avg_points": np.nan,
            "exact_hit_rate": np.nan,
            "goal_diff_hit_rate": np.nan,
            "outcome_hit_rate": np.nan,
            "miss_rate": np.nan,
        }

    pred_home = np.rint(data[pred_home_col].to_numpy(dtype=float)).astype(int)
    pred_away = np.rint(data[pred_away_col].to_numpy(dtype=float)).astype(int)
    actual_home = np.rint(data[actual_home_col].to_numpy(dtype=float)).astype(int)
    actual_away = np.rint(data[actual_away_col].to_numpy(dtype=float)).astype(int)

    points = np.array(
        [
            score_single_prediction(
                pred_home=int(ph),
                pred_away=int(pa),
                actual_home=int(ah),
                actual_away=int(aa),
                rule=scoring_rule,
            )
            for ph, pa, ah, aa in zip(pred_home, pred_away, actual_home, actual_away)
        ],
        dtype=int,
    )

    pred_diff = pred_home - pred_away
    actual_diff = actual_home - actual_away
    exact_mask = (pred_home == actual_home) & (pred_away == actual_away)
    goal_diff_mask = (pred_diff == actual_diff) & ~exact_mask
    outcome_mask = (np.sign(pred_diff) == np.sign(actual_diff)) & ~exact_mask & ~goal_diff_mask
    miss_mask = ~(exact_mask | goal_diff_mask | outcome_mask)

    matches = int(points.size)
    total_points = int(points.sum())
    avg_points = float(points.mean())

    return {
        "matches_evaluated": matches,
        "total_points": total_points,
        "avg_points": avg_points,
        "exact_hit_rate": float(exact_mask.mean()),
        "goal_diff_hit_rate": float(goal_diff_mask.mean()),
        "outcome_hit_rate": float(outcome_mask.mean()),
        "miss_rate": float(miss_mask.mean()),
    }
