"""Composable model components for probability matrices and optimization.

Public API:
- ProbabilityMatrixBuilder
- PoissonMatrixBuilder
- ExpectedPointsRule
- ScoreOptimizer
- ExpectedPointsOptimizer
"""

from .matrix_builders import PoissonMatrixBuilder, ProbabilityMatrixBuilder
from .optimizers import ExpectedPointsOptimizer, ExpectedPointsRule, ScoreOptimizer

__all__ = [
    "ProbabilityMatrixBuilder",
    "PoissonMatrixBuilder",
    "ExpectedPointsRule",
    "ScoreOptimizer",
    "ExpectedPointsOptimizer",
]
