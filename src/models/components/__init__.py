"""Composable model components for probability matrices and optimization.

Public API:
- ProbabilityMatrixBuilder
- PoissonMatrixBuilder
- RhoCalibrationResult
- calibrate_rho
- plot_rho_calibration
- ExpectedPointsRule
- ScoreOptimizer
- ExpectedPointsOptimizer
"""

from .matrix_builders import (
    PoissonMatrixBuilder,
    ProbabilityMatrixBuilder,
    RhoCalibrationResult,
    calibrate_rho,
    plot_rho_calibration,
)
from .optimizers import ExpectedPointsOptimizer, ExpectedPointsRule, ScoreOptimizer

__all__ = [
    "ProbabilityMatrixBuilder",
    "PoissonMatrixBuilder",
    "RhoCalibrationResult",
    "calibrate_rho",
    "plot_rho_calibration",
    "ExpectedPointsRule",
    "ScoreOptimizer",
    "ExpectedPointsOptimizer",
]
