---
title: src.models.tuning
summary: Grid search (predictive i sezonowy trainable) oraz walk-forward po sezonach
sidebar_title: src.models.tuning
order: 4
description: API modulu src.models.tuning - run_predictive_grid_search, run_predictive_nll_grid_search, run_trainable_grid_search_three_way, make_season_walk_forward_splits.
keywords: src.models.tuning, grid search, NLL, walk-forward, season splits, plot grid search
---

# +lucide:sliders-horizontal+ API — `src.models.tuning`

Grid search dla modeli predykcyjnych (non-trainable i trainable) oraz
sezonowe walk-forward splits dla walidacji czasowej.

Zobacz też: [Grid search i tuning](../guides/06-grid-search-and-tuning.md).

---

## Grid search

::: src.models.tuning.GridSearchResult

::: src.models.tuning.build_param_grid

::: src.models.tuning.run_predictive_grid_search

::: src.models.tuning.run_predictive_nll_grid_search

::: src.models.tuning.run_trainable_grid_search_three_way

## Walk-forward splits

::: src.models.tuning.SeasonWalkForwardFold

::: src.models.tuning.make_season_walk_forward_splits

## Wizualizacja wyników grid search

::: src.models.tuning.plot_grid_search_1d

::: src.models.tuning.plot_grid_search_2d

::: src.models.tuning.plot_nll_grid_search_2d
