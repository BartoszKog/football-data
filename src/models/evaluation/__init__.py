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
- PITDistribution
- PITVariant
- PITDiagnosticsResult
- available_pit_variants
- get_pit_variant
- resolve_pit_variants
- build_pit_components
- build_pit_diagnostics
- randomized_pit_replicates_from_components
- summarize_pit_uniformity
- plot_pit_histogram_replicates
- plot_pit_worm_replicates
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
from .pit import (
    PITDiagnosticsResult,
    PITDistribution,
    PITVariant,
    available_pit_variants,
    build_pit_components,
    build_pit_diagnostics,
    get_pit_variant,
    plot_pit_histogram_replicates,
    plot_pit_worm_replicates,
    randomized_pit_replicates_from_components,
    resolve_pit_variants,
    summarize_pit_uniformity,
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
