---
title: Walidacja sezonowa
summary: Train, val i eval po sezonach — przykłady kodu, foldy, trainable grid search
sidebar_title: Walidacja sezonowa
order: 5
description: Sezonowy walk-forward w football-data — przeciek val vs eval, make_season_walk_forward_splits, wymagania na dane i kolejność sezonów.
keywords: walk-forward, season, early stopping, validation, SeasonWalkForwardFold, football-data
---

# +lucide:calendar-range+ Walidacja sezonowa (train / val / eval)

W tuningu modeli trenowalnych (np. XGBoost z `early_stopping_rounds`) potrzebujesz
**dwóch ról** poza treningiem właściwym:

1. **Walidacja w trakcie `fit`** — na niej early stopping decyduje, kiedy przestać
   dodawać drzewa. Ta część danych **widzi** model wielokrotnie wewnątrz jednego
   treningu.
2. **Ewaluacja na ranking** — świeży, trzymany z boku fragment, na którym
   porównujesz **kombinacje hiperparametrów**. Nie może być tym samym zbiorem
   co (1), bo wtedy subtelnie „przeciekasz” informację z val do wyboru
   najlepszego zestawu parametrów.

## Predictive vs trainable

W [`src.models.tuning`](../api/tuning.md) są dwa główne tory:

- **`run_predictive_grid_search`** — modele bez uczenia (np. `PoissonDixonColesModel`);
  nie ma `fit` ani early stopping, więc **inna** historia podziału danych i przecieku.
- **`run_trainable_grid_search_three_way`** — modele z `fit` (np. `XGBoostPoissonModel`);
  tu właśnie **train / val / eval po sezonach** jest potrzebne, żeby ranking
  hiperparametrów nie był liczony na tym samym sezonie, który steruje zatrzymaniem drzew.

## Geometria jednego folda

Niech `seasons_order = [S0, S1, S2, S3, …]` będzie **chronologiczną** listą etykiet
sezonów używaną w `make_season_walk_forward_splits`. Dla kolejnych kroków pętli
(pierwszy krok zaczyna się, gdy masz co najmniej trzy sezony):

- **train** — wszystkie sezony **ściśle wcześniejsze** niż val i eval, czyli
  `S0 … S_{k-2}` przy val = `S_{k-1}` i eval = `S_k`,
- **val** — sezon **bezpośrednio przed** eval (`S_{k-1}`), zwykle przekazywany do
  `fit(..., eval_df=val_df)` pod early stopping,
- **eval** — „bieżący” sezon folda (`S_k`); **tylko** na nim wołasz `predict` i
  liczysz metryki rankingu dla grid searcha.

Przykład dla czterech sezonów: przy kolejności `[2018/2019, 2019/2020, 2020/2021, 2021/2022]`
pierwszy fold ma train = `[2018/2019]`, val = `2019/2020`, eval = `2020/2021`; drugi —
train = `[2018/2019, 2019/2020]`, val = `2020/2021`, eval = `2021/2022`.

## Jak to jest zrobione w projekcie

`make_season_walk_forward_splits` zwraca listę obiektów
[`SeasonWalkForwardFold`](../api/tuning.md): dla każdego folda masz trzy tablice
**indeksów wierszy** NumPy (`train_indices`, `val_indices`, `eval_indices`) oraz
`fold_id`. To te indeksy podaje później m.in. trainable grid search.

- **train** — wszystkie wiersze z sezonów wcześniejszych niż val i eval,
- **val** — wiersze sezonu tuż przed eval,
- **eval** — wiersze sezonu ewaluacji; **tylko** tam metryki rankingu dla
  `run_trainable_grid_search_three_way`.

Dzięki temu ranking hiperparametrów jest liczony na **out-of-sample** w sensie
sezonowym, bez ponownego użycia tego samego sezonu co early stopping.

## +lucide:code+ Przykłady w kodzie

Poniższe snippety są świadomie zbliżone do
[sekcji 6 w Grid search i tuning](../guides/06-grid-search-and-tuning.md#6-trainable-grid-search-sezonowy-walk-forward),
żeby widać było **co wołać, z czym i po co**.

### Budowa foldów i ręczne `train` / `val` / `eval`

`SeasonWalkForwardFold` trzyma **pozycje wierszy** w `historical_df`. Z jednego
folda robisz trzy ramki przez `iloc`:

```python
from src.models.tuning import make_season_walk_forward_splits

historical_seasons = ["2018/2019", "2019/2020", "2020/2021", "2021/2022"]
historical_df = df[df["season"].isin(historical_seasons)].copy()

folds = make_season_walk_forward_splits(
    historical_df,
    season_col="season",
    seasons_order=historical_seasons,
)

fold = folds[0]
train_df = historical_df.iloc[fold.train_indices]
val_df = historical_df.iloc[fold.val_indices]
eval_df = historical_df.iloc[fold.eval_indices]

# val_df:
#   fit(..., eval_df=val_df)
#   (np. early stopping XGBoost)
# eval_df:
#   predict + metryki rankingu hiperparametrów
#   (osobny sezon niż val)
```

Dla `folds[0]` przy powyższej kolejności: `train_df` to tylko `2018/2019`,
`val_df` to `2019/2020`, `eval_df` to `2020/2021`. Kolejny element listy
`folds` przesuwa to okno o jeden sezon w przód.

### Trainable grid search (ten sam podział, automatyczny ranking)

Zamiast ręcznie pętlić po foldach i parametrach, `run_trainable_grid_search_three_way`
dla każdej kombinacji hiperparametrów i każdego folda robi wewnętrznie
`fit` na `train_df` z `eval_df=val_df`, a **metryki rankingu** liczy na
`eval_df`. Musisz mieć zdefiniowaną fabrykę modelu (jak w przewodniku) oraz
`historical_df` z kolumnami cech i celów pod `XGBoostPoissonModel`.

```python
from src.models.tuning import (
    make_season_walk_forward_splits,
    run_trainable_grid_search_three_way,
)

historical_seasons = ["2018/2019", "2019/2020", "2020/2021", "2021/2022"]
historical_df = df[df["season"].isin(historical_seasons)].copy()

folds = make_season_walk_forward_splits(
    historical_df,
    season_col="season",
    seasons_order=historical_seasons,
)

search = run_trainable_grid_search_three_way(
    model_factory=trainable_model_factory,
    param_grid={"learning_rate": [0.02, 0.025, 0.03]},
    df=historical_df,
    folds=folds,
    datetime_col="match_date",
    score_key="avg_points",
    cache_mode="off",
)
```

`trainable_model_factory` to callable zwracający świeży `XGBoostPoissonModel`
(dokładny wzorzec i siatka parametrów: przewodnik powyżej). Dla modeli
**bez** `fit` używasz `run_predictive_grid_search` — inny przepływ, bez `val`
pod early stopping.

## Przygotowanie `DataFrame`

Zanim zbudujesz splitty, `df` powinien zawierać już **tylko sezony historyczne**,
bez holdoutów w stylu `current` — inaczej „przyszła” runda zmiesza się z
walk-forward. Patrz: [Sezony i holdout](../guides/01-loading-data.md#sezony-i-holdout).

## Parametr `seasons_order`

!!! warning "Kolejność sezonów musi być świadoma"
    Gdy **nie** podasz `seasons_order`, funkcja bierze unikalne wartości `season`
    w **kolejności pierwszego wystąpienia w `df`** — to nie zawsze jest sens
    chronologiczny. Przy walidacji sezonowej **zawsze** ustaw `seasons_order`
    explicite (np. lista posortowana według roku lub znanego porządku rozgrywek).

Potrzebujesz **co najmniej trzech** sezonów w `seasons_order`; w przeciwnym razie
funkcja rzuca `ValueError`.

Jeśli któryś fold miałby pusty train, val lub eval, przy `strict=False` (domyślnie)
taki krok jest **pomijany** z ostrzeżeniem; przy `strict=True` — błąd. Szczegóły
parametrów: [`make_season_walk_forward_splits`](../api/tuning.md).

!!! tip "Pełny kontekst (fabryka, metryki, cache)"
    Rozszerzona narracja, tabela predictive vs trainable i odsyłacze do API:
    [Grid search i tuning — sekcja 6 (Trainable)](../guides/06-grid-search-and-tuning.md#6-trainable-grid-search-sezonowy-walk-forward).

## Zobacz też

- [Grid search i tuning — sekcja 6 (Trainable)](../guides/06-grid-search-and-tuning.md#6-trainable-grid-search-sezonowy-walk-forward)
- [API `src.models.tuning`](../api/tuning.md)
- [Ewaluacja predykcji](../guides/05-evaluating-predictions.md)
