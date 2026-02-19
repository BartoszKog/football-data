"""Modeling package for football match score prediction and evaluation.

Public API:
- PoissonDixonColesModel
- PredictiveModel
- TrainablePredictiveModel
- ScoreRule
- score_single_prediction
- evaluate_score_predictions
- GridSearchResult
- build_param_grid
- run_predictive_grid_search
- TimeSeriesFold
- make_walk_forward_splits
- run_trainable_grid_search
- plot_grid_search_1d
- plot_grid_search_2d
"""

from .evaluation import ScoreRule, evaluate_score_predictions, score_single_prediction
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
    "evaluate_score_predictions",
    "GridSearchResult",
    "TimeSeriesFold",
    "build_param_grid",
    "make_walk_forward_splits",
    "run_predictive_grid_search",
    "run_trainable_grid_search",
    "plot_grid_search_1d",
    "plot_grid_search_2d",
]
