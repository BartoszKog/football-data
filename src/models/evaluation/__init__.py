"""Evaluation helpers for scoreline prediction models.

Public API:
- ScoreRule
- score_single_prediction
- compute_points_per_match
- evaluate_score_predictions
- evaluate_poisson_deviance
- compare_deviance_paired_ttest
- PointsSummary1x2
- summarize_predictions_1x2
- plot_predictions_summary
"""

from .scoring import (
    ScoreRule,
    compare_deviance_paired_ttest,
    compute_points_per_match,
    evaluate_poisson_deviance,
    evaluate_score_predictions,
    score_single_prediction,
)
from .visualization import (
    PointsSummary1x2,
    plot_predictions_summary,
    summarize_predictions_1x2,
)

__all__ = [
    "ScoreRule",
    "score_single_prediction",
    "compute_points_per_match",
    "evaluate_score_predictions",
    "evaluate_poisson_deviance",
    "compare_deviance_paired_ttest",
    "PointsSummary1x2",
    "summarize_predictions_1x2",
    "plot_predictions_summary",
]
