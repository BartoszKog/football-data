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
    """Select scoreline maximizing expected points under given rules.

    The optimizer uses a precomputed lookup tensor to avoid per-cell loops
    in ``optimize``. For prediction indices ``(ph, pa)`` and real score
    indices ``(rh, ra)``, lookup value:
    ``_points_matrix[ph, pa, rh, ra]`` stores awarded points.
    """

    def __init__(
        self,
        *,
        rules: ExpectedPointsRule,
        max_goals_prediction: int,
        max_goals_matrix: int,
    ) -> None:
        """Initialize optimizer configuration and points lookup tensor.

        Parameters
        ----------
        rules:
            Scoring rules for exact, goal-difference and outcome hits.
        max_goals_prediction:
            Maximum goals considered in candidate predictions.
        max_goals_matrix:
            Maximum goals represented in the input probability matrix.
        """
        self.rules = rules
        max_goals_prediction_value = int(max_goals_prediction)
        max_goals_matrix_value = int(max_goals_matrix)
        if max_goals_prediction_value < 0:
            raise ValueError("max_goals_prediction cannot be negative.")
        if max_goals_matrix_value < 0:
            raise ValueError("max_goals_matrix cannot be negative.")
        if max_goals_prediction_value > max_goals_matrix_value:
            raise ValueError("max_goals_prediction cannot exceed max_goals_matrix.")
        self.max_goals_prediction = max_goals_prediction_value
        self.max_goals_matrix = max_goals_matrix_value
        self._points_matrix = self._build_points_lookup()

    def optimize(self, matrix: np.ndarray) -> tuple[int, int, float]:
        """Return scoreline with highest expected points.

        This method is fully vectorized:
        - expected points for all predictions are computed with one
          ``np.tensordot`` call,
        - ties on expected points (``np.isclose``) are broken by choosing
          the candidate with highest direct hit probability from ``matrix``.
          If still tied, ``np.argmax`` selects the first index in row-major
          order (lowest home goals, then lowest away goals).
        """
        if matrix.ndim != 2:
            raise ValueError("matrix must be a 2D array.")
        rows, cols = matrix.shape
        if rows == 0 or cols == 0:
            raise ValueError("matrix must be non-empty.")
        expected_size = self.max_goals_matrix + 1
        if rows != expected_size or cols != expected_size:
            raise ValueError(
                "matrix dimensions must match configured max_goals_matrix."
            )

        # Contract over (real_home, real_away) axes:
        # result shape -> (pred_home, pred_away).
        expected_points = np.tensordot(
            self._points_matrix,
            matrix,
            axes=([2, 3], [0, 1]),
        )
        best_xpts = float(np.max(expected_points))
        best_mask = np.isclose(expected_points, best_xpts)

        prediction_size = self.max_goals_prediction + 1
        direct_prob = matrix[:prediction_size, :prediction_size]
        # Keep only the tied candidates, then pick max direct probability.
        tie_break_values = np.where(best_mask, direct_prob, -np.inf)
        best_index = int(np.argmax(tie_break_values))
        best_home, best_away = np.unravel_index(best_index, tie_break_values.shape)
        return int(best_home), int(best_away), float(expected_points[best_home, best_away])

    def _build_points_lookup(self) -> np.ndarray:
        """Build 4D lookup tensor for expected-points scoring.

        Tensor dimensions are:
        ``(pred_home, pred_away, real_home, real_away)``.
        """
        prediction_size = self.max_goals_prediction + 1
        matrix_size = self.max_goals_matrix + 1

        pred_home = np.arange(prediction_size)[:, None, None, None]
        pred_away = np.arange(prediction_size)[None, :, None, None]
        real_home = np.arange(matrix_size)[None, None, :, None]
        real_away = np.arange(matrix_size)[None, None, None, :]

        pred_diff = pred_home - pred_away
        real_diff = real_home - real_away

        exact_mask = (pred_home == real_home) & (pred_away == real_away)
        goal_diff_mask = pred_diff == real_diff
        outcome_mask = np.sign(pred_diff) == np.sign(real_diff)

        points = np.full(
            (prediction_size, prediction_size, matrix_size, matrix_size),
            self.rules.miss,
            dtype=float,
        )
        points[outcome_mask] = self.rules.outcome
        points[goal_diff_mask] = self.rules.goal_diff
        points[exact_mask] = self.rules.exact
        return points
