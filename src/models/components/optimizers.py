"""Score prediction optimizers used by statistical and ML models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class ScoreOptimizer(Protocol):
    """Protocol for selecting scoreline from probability matrix."""

    def optimize(self, matrix: np.ndarray) -> tuple[int, int, float]:
        """Return best home goals, away goals and expected points."""


@dataclass(frozen=True)
class ExpectedPointsRule:
    """Point rule used for expected points maximization."""

    exact: int = 3
    goal_diff: int = 2
    outcome: int = 1
    miss: int = 0


class ExpectedPointsOptimizer(ScoreOptimizer):
    """Select scoreline maximizing expected points under given rules."""

    def __init__(
        self,
        *,
        rules: ExpectedPointsRule,
        max_goals_prediction: int,
    ) -> None:
        """Initialize optimizer configuration."""
        self.rules = rules
        max_goals_prediction_value = int(max_goals_prediction)
        if max_goals_prediction_value < 0:
            raise ValueError("max_goals_prediction cannot be negative.")
        self.max_goals_prediction = max_goals_prediction_value

    def optimize(self, matrix: np.ndarray) -> tuple[int, int, float]:
        """Return scoreline with highest expected points."""
        if matrix.ndim != 2:
            raise ValueError("matrix must be a 2D array.")
        rows, cols = matrix.shape
        if rows == 0 or cols == 0:
            raise ValueError("matrix must be non-empty.")
        if self.max_goals_prediction >= rows or self.max_goals_prediction >= cols:
            raise ValueError(
                "max_goals_prediction must fit within matrix dimensions."
            )

        best_home = 0
        best_away = 0
        best_xpts = -np.inf
        best_point_prob = -np.inf

        for pred_home in range(self.max_goals_prediction + 1):
            for pred_away in range(self.max_goals_prediction + 1):
                expected_points = self._expected_points(
                    pred_home=pred_home,
                    pred_away=pred_away,
                    matrix=matrix,
                )
                direct_prob = float(matrix[pred_home, pred_away])
                if (
                    expected_points > best_xpts
                    or (
                        np.isclose(expected_points, best_xpts)
                        and direct_prob > best_point_prob
                    )
                ):
                    best_xpts = expected_points
                    best_point_prob = direct_prob
                    best_home = pred_home
                    best_away = pred_away

        return best_home, best_away, float(best_xpts)

    def _expected_points(
        self,
        *,
        pred_home: int,
        pred_away: int,
        matrix: np.ndarray,
    ) -> float:
        expected = 0.0
        rows, cols = matrix.shape
        for real_home in range(rows):
            for real_away in range(cols):
                probability = float(matrix[real_home, real_away])
                points = self._score_points(
                    pred_home=pred_home,
                    pred_away=pred_away,
                    real_home=real_home,
                    real_away=real_away,
                )
                expected += probability * points
        return float(expected)

    def _score_points(
        self,
        *,
        pred_home: int,
        pred_away: int,
        real_home: int,
        real_away: int,
    ) -> int:
        if pred_home == real_home and pred_away == real_away:
            return self.rules.exact

        pred_diff = pred_home - pred_away
        real_diff = real_home - real_away
        if pred_diff == real_diff:
            return self.rules.goal_diff

        pred_outcome = int(np.sign(pred_diff))
        real_outcome = int(np.sign(real_diff))
        if pred_outcome == real_outcome:
            return self.rules.outcome
        return self.rules.miss
