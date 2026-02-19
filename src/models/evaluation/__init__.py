"""Evaluation helpers for scoreline prediction models.

Public API:
- ScoreRule
- score_single_prediction
- evaluate_score_predictions
"""

from .scoring import ScoreRule, evaluate_score_predictions, score_single_prediction

__all__ = [
    "ScoreRule",
    "score_single_prediction",
    "evaluate_score_predictions",
]
