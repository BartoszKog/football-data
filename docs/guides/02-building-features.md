---
title: Budowa cech
summary: Prawdopodobieństwa z kursów i lambdy Poissona — src.features
sidebar_title: Budowa cech
order: 2
description: Pipeline prawdopodobieństw implikowanych i lambd Poissona dla modeli football-data.
keywords: src.features, implied probabilities, poisson lambda, trimmed_avg
---

# +lucide:layers+ Budowa cech

Po [załadowaniu danych](01-loading-data.md) zwykle dodajesz:

1. **Prawdopodobieństwa implikowane** z kursów (metoda potęgowa).
2. **Lambdy Poissona** — baseline z 1X2 + over 2.5, potem opcjonalnie wersja skalibrowana.

Pełne API: [`src.features`](../api/features.md).

## 1. Prawdopodobieństwa

Najszybciej — standardowe rynki (1x2, BTTS, over/under 2.5) z jednym prefiksem kursów:

```python
from src.features import add_power_implied_probabilities_standard_markets

df = add_power_implied_probabilities_standard_markets(
    df,
    odds_prefix="trimmed_avg",
    output_prefix="prob_trimmed_avg",
    errors="coerce",
)
```

Inny prefiks (np. `max`): ten sam helper z `odds_prefix="max"`.

Intuicja metody: [Kursy → prawdopodobieństwa](../concepts/odds-to-probabilities.md).

## 2. Lambdy baseline i skalibrowane

Baseline z kolumn prawdopodobieństw 1X2 i over 2.5:

```python
from src.features import add_baseline_poisson_lambdas

df = add_baseline_poisson_lambdas(
    df,
    prob_home_col="prob_trimmed_avg_1",
    prob_away_col="prob_trimmed_avg_2",
    prob_over25_col="prob_trimmed_avg_over_25",
    bias_correction=1.035,
)
```

Kalibracja względem baseline (domyślne współczynniki z GAM lab):

```python
from src.features import add_calibrated_poisson_lambdas

df = add_calibrated_poisson_lambdas(df)
```

Teoria: [Lambdy Poissona](../concepts/poisson-lambdas.md).

## 3. Przykład pełnego łańcucha

```python
from src.features import (
    add_power_implied_probabilities_standard_markets,
    add_calibrated_poisson_lambdas,
)

df = (
    df.pipe(
        add_power_implied_probabilities_standard_markets,
        odds_prefix="trimmed_avg",
        output_prefix="prob_trimmed_avg",
    ).pipe(add_calibrated_poisson_lambdas)
)
```

## Skrót kolumn (orientacyjnie)

| Krok | Przykładowe nowe kolumny |
| --- | --- |
| Power implied | `prob_trimmed_avg_1`, `prob_trimmed_avg_X`, `prob_trimmed_avg_2`, … |
| Baseline λ | `baseline_lambda_home`, `baseline_lambda_away` |
| Calibrated λ | `calibrated_lambda_home`, `calibrated_lambda_away` |

## Zobacz też

- [Trening Poisson Dixon–Coles](03-training-poisson-dc.md)
- [Trening XGBoost Poisson](04-training-xgboost.md)
