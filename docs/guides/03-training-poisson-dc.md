---
title: Poisson Dixon–Coles
summary: PoissonDixonColesModel — kolumny wejściowe, wynik predict, ewaluacja
sidebar_title: Poisson Dixon–Coles
order: 3
description: Jak uruchomić model Poissona z korektą Dixona-Colesa i ocenić predykcje w football-data.
keywords: PoissonDixonColesModel, dixon-coles, rho, predict, football-data
---

# +lucide:chart-line+ Trening i predykcja Poisson Dixon–Coles

`PoissonDixonColesModel` jest modelem **bez uczenia** (non-trainable): ma
parametry jak `rho` i `bias_correction`, które ustawiasz z zewnątrz lub
stroisz grid searchem.

API: [`PoissonDixonColesModel`](../api/models.md), korekta \(\rho\):
[Korekta Dixona-Colesa](../concepts/dixon-coles-correction.md).

## Minimalny przykład

Wymagane są kolumny prawdopodobieństw (np. z [Budowa cech](02-building-features.md)):

```python
from src.models import PoissonDixonColesModel, evaluate_score_predictions

model = PoissonDixonColesModel(
    prob_home_col="prob_trimmed_avg_1",
    prob_away_col="prob_trimmed_avg_2",
    prob_over25_col="prob_trimmed_avg_over_25",
    rho=-0.06,
    bias_correction=1.05,
    use_over25_interpolation=True,
)

pred_df = model.predict(df)

metrics = evaluate_score_predictions(
    pred_df,
    pred_home_col="pred_home_goals",
    pred_away_col="pred_away_goals",
    actual_home_col="home_score",
    actual_away_col="away_score",
)
```

Nazwy kolumn `prob_*` muszą odpowiadać temu, co masz w `df` — możesz użyć
innego prefiksu niż `prob_trimmed_avg_*`.

## +lucide:table-2+ Wynik `predict`: jeden `DataFrame`, dopisane kolumny

`model.predict(df)` **nie zwraca osobnej „ramki predykcji”** — zwraca **kopię**
`df` z **doklejonymi z prawej strony** kolumnami wynikowymi (indeks jak w
wejściu). To ten sam obiekt co `pd.concat([df, output], axis=1)` w implementacji.

Dopisane kolumny (nazwy po angielsku, jak w kodzie):

| Kolumna | Typ / sens |
| --- | --- |
| `pred_home_goals` | Całkowita liczba bramek gospodarza wybrana przez optymalizator oczekiwanych punktów. |
| `pred_away_goals` | Jak wyżej dla gościa. |
| `pred_score` | Tekst w formacie `"h:a"` (np. `"2:1"`) — spójny z `pred_home_goals` / `pred_away_goals`. |
| `pred_xpts` | Oczekiwana liczba punktów (Supertyper-like) dla wybranego wyniku przy zadanej macierzy prawdopodobieństw wyniku — [jak to działa](../concepts/expected-points-optimization.md). |
| `exp_goals_home` | Oczekiwana liczba bramek gospodarza \(\lambda_{home}\) po mapowaniu `prob_over25` → suma bramek, podziale proporcji `prob_home` / `prob_away` i `bias_correction` — **to** wchodzi w budowę macierzy Poissona (z korektą \(\rho\)). |
| `exp_goals_away` | Analogicznie \(\lambda_{away}\). |

Przy `errors="coerce"` (domyślnie) wiersze z niepoprawnymi prawdopodobieństwami
dostają w tych kolumnach wartości puste (`pd.NA`), zamiast rzucać wyjątkiem.

## Dobór `rho` i innych hiperparametrów

- Grid search 1D/2D po `avg_points`: [Grid search i tuning](06-grid-search-and-tuning.md).
- Kalibracja vs NLL: [Korekta Dixona-Colesa](../concepts/dixon-coles-correction.md).

## Kontrakt interfejsu

```python
from src.models import PredictiveModel, PoissonDixonColesModel

model: PredictiveModel = PoissonDixonColesModel(rho=-0.06)
pred_df = model.predict(df)
```

## Zobacz też

- [Ewaluacja predykcji](05-evaluating-predictions.md)
- [Zasady punktacji](../concepts/scoring-rules.md)
