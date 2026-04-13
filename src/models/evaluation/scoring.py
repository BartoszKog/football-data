"""Reusable scoreline evaluation helpers for prediction models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ttest_rel


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


def compute_points_per_match(
    df: pd.DataFrame,
    *,
    pred_home_col: str = "pred_home_goals",
    pred_away_col: str = "pred_away_goals",
    actual_home_col: str = "home_score",
    actual_away_col: str = "away_score",
    rule: ScoreRule | None = None,
) -> pd.Series:
    """Return points per match for scoreline predictions.

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
    pd.Series
        Points per row (int), index aligned with valid rows after dropna.
        Rows with NaN in required columns are excluded.
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
        return pd.Series(dtype=int)

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

    return pd.Series(points, index=data.index, dtype=int)


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


# ---------------------------------------------------------------------------
# Poisson-deviance evaluation
# ---------------------------------------------------------------------------


def _poisson_deviance_per_sample(
    y_true: np.ndarray, y_pred: np.ndarray
) -> np.ndarray:
    """Per-observation Poisson deviance (same formula as sklearn ``mean_poisson_deviance``)."""
    y_true = np.asarray(y_true, dtype=np.float64).ravel()
    y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64).ravel(), 1e-15)
    safe_y_true = np.where(y_true > 0, y_true, 1.0)
    term1 = np.where(y_true > 0, y_true * np.log(safe_y_true / y_pred), 0.0)
    term2 = y_true - y_pred
    return 2.0 * (term1 - term2)


def evaluate_poisson_deviance(
    y_true_home: np.ndarray,
    y_pred_home: np.ndarray,
    y_true_away: np.ndarray,
    y_pred_away: np.ndarray,
) -> dict[str, Any]:
    """Compute per-match Poisson deviance for home and away goal predictions.

    Parameters
    ----------
    y_true_home, y_true_away:
        Observed goal counts (one element per match).
    y_pred_home, y_pred_away:
        Predicted expected goals / Poisson lambda (one element per match).

    Returns
    -------
    dict[str, Any]
        ``Deviance_home``, ``SE_home``, ``Deviance_away``, ``SE_away``,
        ``Deviance_mean``, ``SE_mean`` (rounded to 4 d.p.) and
        ``Error_Vector`` (Python list: per-match home deviances followed by
        away deviances -- usable with :func:`compare_deviance_paired_ttest`).
    """
    dev_home = _poisson_deviance_per_sample(y_true_home, y_pred_home)
    dev_away = _poisson_deviance_per_sample(y_true_away, y_pred_away)
    n = int(dev_home.shape[0])
    if n == 0:
        raise ValueError("evaluate_poisson_deviance requires at least one match.")
    se_home = float(np.std(dev_home, ddof=1) / np.sqrt(n))
    se_away = float(np.std(dev_away, ddof=1) / np.sqrt(n))
    per_match_mean = (dev_home + dev_away) / 2.0
    se_mean = float(np.std(per_match_mean, ddof=1) / np.sqrt(n))
    err_vec = np.concatenate([dev_home, dev_away])
    return {
        "Deviance_home": round(float(np.mean(dev_home)), 4),
        "SE_home": round(se_home, 4),
        "Deviance_away": round(float(np.mean(dev_away)), 4),
        "SE_away": round(se_away, 4),
        "Deviance_mean": round(float(np.mean(per_match_mean)), 4),
        "SE_mean": round(se_mean, 4),
        "Error_Vector": err_vec.tolist(),
    }


def compare_deviance_paired_ttest(
    current_vector: list[float] | np.ndarray,
    best_vector: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Paired t-test on per-observation deviance vectors (current vs best).

    Lower mean deviance is better.

    Parameters
    ----------
    current_vector, best_vector:
        Flat deviance vectors (typically ``Error_Vector`` from
        :func:`evaluate_poisson_deviance`).
    alpha:
        Significance level for the two-sided test.

    Returns
    -------
    dict[str, Any]
        ``statistic``, ``pvalue``, ``alpha``, ``comparison_status``
        (one of ``"better_significant"``, ``"better_not_significant"``,
        ``"worse"``, ``"error"``), ``mean_current``, ``mean_best``,
        ``message``.
    """
    a = np.asarray(current_vector, dtype=float).ravel()
    b = np.asarray(best_vector, dtype=float).ravel()
    if a.shape != b.shape or a.size == 0:
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "alpha": alpha,
            "comparison_status": "error",
            "mean_current": float("nan"),
            "mean_best": float("nan"),
            "message": "Incompatible vector lengths or empty vector.",
        }
    mean_c = float(np.mean(a))
    mean_b = float(np.mean(b))
    better = mean_c < mean_b
    tt = ttest_rel(a, b)
    pval = float(tt.pvalue) if tt.pvalue is not None else float("nan")
    stat = float(tt.statistic)

    if not better:
        return {
            "statistic": stat,
            "pvalue": pval,
            "alpha": alpha,
            "comparison_status": "worse",
            "mean_current": mean_c,
            "mean_best": mean_b,
            "message": (
                "Current model does not have lower mean Poisson deviance "
                "than the best historical record."
            ),
        }
    if pval < alpha:
        return {
            "statistic": stat,
            "pvalue": pval,
            "alpha": alpha,
            "comparison_status": "better_significant",
            "mean_current": mean_c,
            "mean_best": mean_b,
            "message": (
                "Current model is significantly better (p < alpha) than "
                "the best historical record."
            ),
        }
    return {
        "statistic": stat,
        "pvalue": pval,
        "alpha": alpha,
        "comparison_status": "better_not_significant",
        "mean_current": mean_c,
        "mean_best": mean_b,
        "message": (
            "Mean deviance is lower than the best historical record, "
            "but the difference is not statistically significant."
        ),
    }
