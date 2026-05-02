---
title: src.models
summary: Modele predykcyjne (Poisson DC, XGBoost Poisson) i wspolne komponenty
sidebar_title: src.models
order: 3
description: API modulu src.models - PoissonDixonColesModel, XGBoostPoissonModel, kontrakty interfejsow, komponenty wspolne.
keywords: src.models, poisson, dixon-coles, xgboost, matrix builder, expected points
---

# +lucide:package+ API — `src.models`

Modele predykcyjne (statystyczne i ML) pod wspólnym kontraktem
[`PredictiveModel`][src.models.interfaces.PredictiveModel] /
[`TrainablePredictiveModel`][src.models.interfaces.TrainablePredictiveModel].

Zobacz też: [Korekta Dixona-Colesa](../concepts/dixon-coles-correction.md),
[Grid search i tuning](../guides/06-grid-search-and-tuning.md).

---

## Kontrakty interfejsów

::: src.models.interfaces.PredictiveModel

::: src.models.interfaces.TrainablePredictiveModel

## Modele statystyczne

::: src.models.statistical.PoissonDixonColesModel

## Modele ML

::: src.models.ml.XGBoostPoissonModel

## Komponenty wspólne

::: src.models.components.PoissonMatrixBuilder

::: src.models.components.ProbabilityMatrixBuilder

::: src.models.components.RhoCalibrationResult

::: src.models.components.calibrate_rho

::: src.models.components.average_scoreline_nll

::: src.models.components.average_points_weighted_scoreline_nll

::: src.models.components.plot_rho_calibration

::: src.models.components.ExpectedPointsRule

::: src.models.components.ScoreOptimizer

::: src.models.components.ExpectedPointsOptimizer
