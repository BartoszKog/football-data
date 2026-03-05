"""Probability matrix builders used by score prediction models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np
from scipy.stats import poisson


@runtime_checkable
class ProbabilityMatrixBuilder(Protocol):
    """Protocol for building scoreline probability matrices."""

    def build_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """Build and return a normalized scoreline probability matrix."""


class PoissonMatrixBuilder(ProbabilityMatrixBuilder):
    """Build Poisson scoreline matrix with Dixon-Coles low-score correction.

    The implementation is vectorized for backtesting workloads:
    - one goals vector is created with ``np.arange``,
    - home and away PMF vectors are evaluated once each,
    - the base matrix is computed with ``np.outer``,
    - Dixon-Coles correction is applied only to the four low-score cells.
    """

    def __init__(self, *, rho: float, max_goals_matrix: int) -> None:
        """Initialize Poisson matrix builder configuration."""
        rho_value = float(rho)
        max_goals_value = int(max_goals_matrix)
        if not np.isfinite(rho_value):
            raise ValueError("rho must be finite.")
        if max_goals_value < 2:
            raise ValueError("max_goals_matrix must be at least 2.")
        self.rho = rho_value
        self.max_goals_matrix = max_goals_value

    def build_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """Build normalized scoreline probability matrix.

        Parameters
        ----------
        lambda_home:
            Expected goals for the home team.
        lambda_away:
            Expected goals for the away team.

        Returns
        -------
        np.ndarray
            A 2D matrix of shape ``(max_goals_matrix + 1, max_goals_matrix + 1)``
            where ``matrix[i, j]`` is the probability of score ``i:j``.

        Notes
        -----
        Dixon-Coles correction is only needed for outcomes:
        ``(0,0)``, ``(0,1)``, ``(1,0)``, ``(1,1)``.
        """
        if not np.isfinite(lambda_home) or not np.isfinite(lambda_away):
            raise ValueError("lambda_home and lambda_away must be finite.")
        if lambda_home < 0 or lambda_away < 0:
            raise ValueError("lambda_home and lambda_away must be non-negative.")
        size = self.max_goals_matrix + 1
        goals = np.arange(size)
        p_home = poisson.pmf(goals, lambda_home)
        p_away = poisson.pmf(goals, lambda_away)
        # Independent Poisson assumptions: P(X=i, Y=j) = P(X=i) * P(Y=j).
        matrix = np.outer(p_home, p_away)

        # Dixon-Coles low-score correction affects only these four outcomes.
        matrix[0, 0] *= 1.0 - (lambda_home * lambda_away * self.rho)
        matrix[0, 1] *= 1.0 + (lambda_home * self.rho)
        matrix[1, 0] *= 1.0 + (lambda_away * self.rho)
        matrix[1, 1] *= 1.0 - self.rho

        matrix = np.clip(matrix, 0.0, None)
        matrix_sum = float(matrix.sum())
        if matrix_sum <= 0:
            raise ValueError("probability matrix sum is not positive.")
        return matrix / matrix_sum
