"""Modeling package for football match score prediction and evaluation.

Public API:
- PoissonDixonColesModel
- PredictiveModel
- TrainablePredictiveModel
- ScoreRule
- score_single_prediction
- compute_points_per_match
- evaluate_score_predictions
- PointsSummary1x2
- summarize_predictions_1x2
- plot_predictions_summary
- GridSearchResult
- build_param_grid
- run_predictive_grid_search
- TimeSeriesFold
- make_walk_forward_splits
- run_trainable_grid_search
- plot_grid_search_1d
- plot_grid_search_2d
"""

from .evaluation import (
    PointsSummary1x2,
    ScoreRule,
    compute_points_per_match,
    evaluate_score_predictions,
    plot_predictions_summary,
    score_single_prediction,
    summarize_predictions_1x2,
)
from .interfaces import PredictiveModel, TrainablePredictiveModel
from .statistical import PoissonDixonColesModel
from .tuning import (
    GridSearchResult,
    TimeSeriesFold,
    build_param_grid,
    make_walk_forward_splits,
    plot_grid_search_1d,
    plot_grid_search_2d,
    run_predictive_grid_search,
    run_trainable_grid_search,
)

__all__ = [
    "PoissonDixonColesModel",
    "PredictiveModel",
    "TrainablePredictiveModel",
    "ScoreRule",
    "score_single_prediction",
    "compute_points_per_match",
    "evaluate_score_predictions",
    "PointsSummary1x2",
    "summarize_predictions_1x2",
    "plot_predictions_summary",
    "GridSearchResult",
    "TimeSeriesFold",
    "build_param_grid",
    "make_walk_forward_splits",
    "run_predictive_grid_search",
    "run_trainable_grid_search",
    "plot_grid_search_1d",
    "plot_grid_search_2d",
]
