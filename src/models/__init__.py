"""Modeling package for football match score prediction and evaluation.

Public API:
- XGBoostPoissonModel
- PoissonDixonColesModel
- PredictiveModel
- TrainablePredictiveModel
- ScoreRule
- score_single_prediction
- compute_points_per_match
- evaluate_score_predictions
- evaluate_poisson_deviance
- compare_deviance_paired_ttest
- PointsSummary1x2
- summarize_predictions_1x2
- plot_predictions_summary
- GridSearchResult
- build_param_grid
- run_predictive_grid_search
- plot_grid_search_1d
- plot_grid_search_2d
- RhoCalibrationResult
- calibrate_rho
- plot_rho_calibration
"""

from .evaluation import (
    PointsSummary1x2,
    ScoreRule,
    compare_deviance_paired_ttest,
    compute_points_per_match,
    evaluate_poisson_deviance,
    evaluate_score_predictions,
    plot_predictions_summary,
    score_single_prediction,
    summarize_predictions_1x2,
)
from .components import RhoCalibrationResult, calibrate_rho, plot_rho_calibration
from .interfaces import PredictiveModel, TrainablePredictiveModel
from .ml import XGBoostPoissonModel
from .statistical import PoissonDixonColesModel
from .tuning import (
    GridSearchResult,
    build_param_grid,
    plot_grid_search_1d,
    plot_grid_search_2d,
    run_predictive_grid_search,
)

__all__ = [
    "XGBoostPoissonModel",
    "PoissonDixonColesModel",
    "PredictiveModel",
    "TrainablePredictiveModel",
    "ScoreRule",
    "score_single_prediction",
    "compute_points_per_match",
    "evaluate_score_predictions",
    "evaluate_poisson_deviance",
    "compare_deviance_paired_ttest",
    "PointsSummary1x2",
    "summarize_predictions_1x2",
    "plot_predictions_summary",
    "GridSearchResult",
    "build_param_grid",
    "run_predictive_grid_search",
    "plot_grid_search_1d",
    "plot_grid_search_2d",
    "RhoCalibrationResult",
    "calibrate_rho",
    "plot_rho_calibration",
]
