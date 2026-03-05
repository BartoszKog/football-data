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
    """Build Poisson scoreline matrix with Dixon-Coles low-score correction."""

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
        """Build normalized probability matrix for the provided goal rates."""
        if not np.isfinite(lambda_home) or not np.isfinite(lambda_away):
            raise ValueError("lambda_home and lambda_away must be finite.")
        if lambda_home < 0 or lambda_away < 0:
            raise ValueError("lambda_home and lambda_away must be non-negative.")
        size = self.max_goals_matrix + 1
        matrix = np.zeros((size, size), dtype=float)

        for home_goals in range(size):
            p_home = poisson.pmf(home_goals, lambda_home)
            for away_goals in range(size):
                p_away = poisson.pmf(away_goals, lambda_away)
                base_prob = p_home * p_away
                correction = self._dixon_coles_tau(
                    home_goals=home_goals,
                    away_goals=away_goals,
                    lambda_home=lambda_home,
                    lambda_away=lambda_away,
                    rho=self.rho,
                )
                matrix[home_goals, away_goals] = base_prob * correction

        matrix = np.clip(matrix, 0.0, None)
        matrix_sum = float(matrix.sum())
        if matrix_sum <= 0:
            raise ValueError("probability matrix sum is not positive.")
        return matrix / matrix_sum

    @staticmethod
    def _dixon_coles_tau(
        *,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float,
        rho: float,
    ) -> float:
        if home_goals == 0 and away_goals == 0:
            return 1.0 - (lambda_home * lambda_away * rho)
        if home_goals == 0 and away_goals == 1:
            return 1.0 + (lambda_home * rho)
        if home_goals == 1 and away_goals == 0:
            return 1.0 + (lambda_away * rho)
        if home_goals == 1 and away_goals == 1:
            return 1.0 - rho
        return 1.0
