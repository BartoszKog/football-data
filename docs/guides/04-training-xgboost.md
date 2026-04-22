---
title: Trening XGBoost Poisson
summary: XGBoostPoissonModel, cechy train/predict, fit z eval_df, wynik predict
sidebar_title: XGBoost Poisson
order: 4
description: Jak trenować parę regresorów XGBoost Poisson, jakich kolumn wymaga predict i stroić hiperparametry sezonowym grid searchem.
keywords: XGBoostPoissonModel, xgboost, trainable, walk-forward, football-data
---

# +lucide:cpu+ Trening XGBoost Poisson

`XGBoostPoissonModel` implementuje [`TrainablePredictiveModel`](../api/models.md):
najpierw `fit`, potem `predict`. Współdzieli z Poisson DC m.in. parametr `rho`
przy składaniu scoreline.

## Cechy: `features_home` i `features_away`

Parametry **`features_home`** oraz **`features_away`** to **jawny kontrakt**
wejścia do modelu:

- przy **`fit`** z `train_df` (i opcjonalnie `eval_df`) muszą istnieć wszystkie
  te kolumny **oraz** cele `target_home_col` / `target_away_col` (domyślnie
  `home_score`, `away_score`);
- **`model_home`** dostaje macierz `train_df[features_home]` i cel
  `train_df[target_home_col]`;
- **`model_away`** dostaje `train_df[features_away]` i `train_df[target_away_col]`;
- przy **`predict`** w `df` muszą być obecne **wszystkie** kolumny z obu list
  (suma zbiorów — zwykle przekazujesz tę samą listę do obu argumentów, wtedy
  to po prostu ta lista).

Nie ma „ukrytych” cech poza tymi listami: to, czego nie podasz w konstruktorze,
nie trafia ani do treningu, ani do inferencji.

## Cechy (przykładowy zestaw)

Jak w praktyce w notatnikach — ta sama lista kolumn dla obu regresorów
(`features_home=features`, `features_away=features`):

```text
baseline_lambda_home, baseline_lambda_away,
prob_trimmed_avg_1, prob_trimmed_avg_X, prob_trimmed_avg_2,
prob_trimmed_avg_over_25, prob_trimmed_avg_btts_yes,
value_1, value_X, value_2,
```

Najpierw zbuduj te kolumny ([Budowa cech](02-building-features.md) + ewentualne
cechy value z Twojego pipeline’u).

## Fit z early stopping

```python
import xgboost as xgb
from src.models import XGBoostPoissonModel

features = [
    "baseline_lambda_home",
    "baseline_lambda_away",
    "prob_trimmed_avg_1",
    "prob_trimmed_avg_X",
    "prob_trimmed_avg_2",
    "prob_trimmed_avg_over_25",
    "prob_trimmed_avg_btts_yes",
    "value_1",
    "value_X",
    "value_2",
]

model = XGBoostPoissonModel(
    features_home=features,
    features_away=features,
    model_home=xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.025,
        early_stopping_rounds=50,
    ),
    model_away=xgb.XGBRegressor(
        objective="count:poisson",
        n_estimators=1000,
        max_depth=3,
        learning_rate=0.025,
        early_stopping_rounds=50,
    ),
    rho=-0.06,
)

model.fit(train_df, eval_df=val_df)
pred_df = model.predict(test_df)
```

## +lucide:table-2+ Wynik `predict`: kopia `df` z dopisanymi kolumnami

`predict` zwraca **kopię** wejściowego `DataFrame` z **doklejonymi** kolumnami
wynikowymi (ten sam indeks co w `df`), tak jak w Poisson DC — por.
[Wynik `predict` w Poisson Dixon–Coles](03-training-poisson-dc.md#wynik-predict-jeden-dataframe-dopisane-kolumny).

XGBoost najpierw przewiduje ciągłe \(\lambda\) (`model_home.predict` /
`model_away.predict`), potem z macierzy Poissona (z \(\rho\)) wybierany jest
wynik maksymalizujący oczekiwane punkty.

| Kolumna | Sens |
| --- | --- |
| `pred_home_goals` | Bramki gospodarza (domyślna nazwa; można zmienić parametrem `pred_home_col`). |
| `pred_away_goals` | Bramki gościa (`pred_away_col`). |
| `pred_score` | Tekst `"h:a"` zgodny z powyższym. |
| `pred_xpts` | Oczekiwane punkty dla wybranego wyniku przy zadanej macierzy — [jak to działa](../concepts/expected-points-optimization.md). |
| `exp_goals_home` | Surowe \(\lambda\) z regresora gospodarza (przed zaokrągleniem do wyboru wyniku). |
| `exp_goals_away` | Surowe \(\lambda\) z regresora gościa. |

Przy `errors="coerce"` problematyczne wiersze dostają `pd.NA` w kolumnach
wynikowych zamiast przerywać całość.

## Tuning hiperparametrów (sezonowy walk-forward)

Do rankingu kombinacji (`learning_rate`, `rho`, …) używaj
`make_season_walk_forward_splits` +
`run_trainable_grid_search_three_way` — wtedy `val` służy tylko do treningu
(early stopping), a metryki gridu liczone są na **osobnym** sezonie `eval`.

Szczegóły i snippet: [Grid search i tuning — sekcja trainable](06-grid-search-and-tuning.md#6-trainable-grid-search-sezonowy-walk-forward).

Teoria podziału: [Walidacja sezonowa](../concepts/season-walk-forward-validation.md).

## Zobacz też

- [Oczekiwane punkty i wybór wyniku](../concepts/expected-points-optimization.md)
- [Ewaluacja predykcji](05-evaluating-predictions.md)
- [Grid search i tuning](06-grid-search-and-tuning.md)
