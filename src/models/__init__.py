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
- run_predictive_nll_grid_search
- run_predictive_points_weighted_nll_grid_search
- plot_grid_search_1d
- plot_grid_search_2d
- plot_nll_grid_search_2d
- RhoCalibrationResult
- average_scoreline_nll
- calibrate_rho
- plot_rho_calibration
- PITDistribution
- PITVariant
- PITDiagnosticsResult
- build_pit_diagnostics
- plot_pit_histogram_replicates
- plot_pit_worm_replicates
"""

from .evaluation import (
    PointsSummary1x2,
    PITDiagnosticsResult,
    PITDistribution,
    PITVariant,
    ScoreRule,
    available_pit_variants,
    build_pit_components,
    build_pit_diagnostics,
    compare_deviance_paired_ttest,
    compute_points_per_match,
    evaluate_poisson_deviance,
    evaluate_score_predictions,
    get_pit_variant,
    plot_pit_histogram_replicates,
    plot_predictions_summary,
    plot_pit_worm_replicates,
    randomized_pit_replicates_from_components,
    resolve_pit_variants,
    score_single_prediction,
    summarize_pit_uniformity,
    summarize_predictions_1x2,
)
from .components import (
    RhoCalibrationResult,
    average_points_weighted_scoreline_nll,
    average_scoreline_nll,
    calibrate_rho,
    plot_rho_calibration,
)
from .interfaces import PredictiveModel, TrainablePredictiveModel
from .ml import XGBoostPoissonModel
from .statistical import PoissonDixonColesModel
from .tuning import (
    GridSearchResult,
    build_param_grid,
    plot_grid_search_1d,
    plot_grid_search_2d,
    plot_nll_grid_search_2d,
    run_predictive_grid_search,
    run_predictive_nll_grid_search,
    run_predictive_points_weighted_nll_grid_search,
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
    "run_predictive_nll_grid_search",
    "run_predictive_points_weighted_nll_grid_search",
    "plot_grid_search_1d",
    "plot_grid_search_2d",
    "plot_nll_grid_search_2d",
    "RhoCalibrationResult",
    "average_points_weighted_scoreline_nll",
    "average_scoreline_nll",
    "calibrate_rho",
    "plot_rho_calibration",
    "PITDistribution",
    "PITVariant",
    "PITDiagnosticsResult",
    "available_pit_variants",
    "get_pit_variant",
    "resolve_pit_variants",
    "build_pit_components",
    "build_pit_diagnostics",
    "randomized_pit_replicates_from_components",
    "summarize_pit_uniformity",
    "plot_pit_histogram_replicates",
    "plot_pit_worm_replicates",
]
