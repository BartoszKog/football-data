---
title: src.models.evaluation
summary: Punktacja Supertyper, Poisson deviance i wizualizacje predykcji
sidebar_title: src.models.evaluation
order: 5
description: API modulu src.models.evaluation - evaluate_score_predictions, evaluate_poisson_deviance, plot_predictions_summary.
keywords: src.models.evaluation, supertyper, scoring, poisson deviance, paired t-test
---

# +lucide:bar-chart-3+ API — `src.models.evaluation`

Uniwersalna punktacja Supertyper (3/2/1/0), metryki Poisson deviance,
sparowany t-test deviance oraz wizualizacje ewaluacji predykcji.

---

## Punktacja Supertyper

::: src.models.evaluation.ScoreRule

::: src.models.evaluation.score_single_prediction

::: src.models.evaluation.compute_points_per_match

::: src.models.evaluation.evaluate_score_predictions

## Poisson deviance

::: src.models.evaluation.evaluate_poisson_deviance

::: src.models.evaluation.compare_deviance_paired_ttest

## Wizualizacja predykcji

::: src.models.evaluation.PointsSummary1x2

::: src.models.evaluation.summarize_predictions_1x2

::: src.models.evaluation.plot_predictions_summary

example result of `plot_predictions_summary`:

![plot_predictions_summary example](../assets/predictions_summary_poisson_dc.png)
