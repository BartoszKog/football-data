# Football Data

## 📂 Organizacja Projektu

```text
football-data/
├── README.md             <- Główny opis projektu, jak uruchomić, skąd wziąć dane.
├── data/
│   ├── external/         <- Dane zewnętrzne (np. słowniki).
│   ├── raw/              <- Surowe, niezmieniane dane (JSON-y z OddsHarvester).
│   └── processed/        <- Czyste dane gotowe do modelowania (wynik działania skryptów z src).
│
├── notebooks/            <- Notatniki Marimo/Jupyter.
│   ├── exploration/      <- Eksperymenty, sprawdzanie danych, brudnopisy badawcze.
│   ├── features_lab/     <- Notatniki do eksperymentów z cechami.
│   └── reports/          <- Gotowe, "wyczyszczone" analizy prezentujące wyniki (np. dla GitHub Pages).
│
├── outputs/              <- Wygenerowane pliki.
│   ├── figures/          <- Wykresy i grafiki używane w raportach (png, svg).
│   └── reports/          <- Wyeksportowane raporty (np. HTML, PDF).
│
├── sandbox/              <- Kod tymczasowy, testy, skrypty "na raz" (ignorowane przez git).
│
├── scripts/              <- Skrypty uruchomieniowe/automatyzacyjne (np. update_data.py).
│
├── src/                  <- Kod źródłowy projektu (moduły Python).
│   ├── data/             <- Skrypty do pobierania i przetwarzania danych.
│   ├── features/         <- Skrypty do inżynierii cech (np. wyliczanie formy drużyny).
│   ├── models/           <- Modele predykcyjne (ML i statystyczne) oraz ewaluacja.
│   └── visualization/    <- Funkcje pomocnicze do tworzenia wykresów.
│
└── OddsHarvester/        <- Zewnętrzne narzędzie do scrapowania (sub-repozytorium).
```

## Przygotowanie danych (src/data)

Warstwa `src/data` przygotowuje dane do dalszych krokow.

Przykład uzycia w notatniku:

```python
from src.data import load_raw_seasons, add_odds_columns

raw_df = load_raw_seasons("data/raw/1liga_*.json")
df = add_odds_columns(raw_df, trim_drop=1)
```

lub:

```python
from src.data import load_and_add_odds_columns
df = load_and_add_odds_columns(pattern="data/raw/1liga_*.json", trim_drop=1)
```

wersja compact (usuwa surowe kolumny marketow po wyliczeniu kursow):

```python
from src.data import load_and_add_odds_columns_compact
df = load_and_add_odds_columns_compact(pattern="data/raw/1liga_*.json", trim_drop=1)
```

mozna tez wybrac tylko jeden typ kursow, np. tylko `trimmed_avg_*`:

```python
df = load_and_add_odds_columns_compact(
    pattern="data/raw/1liga_*.json",
    trim_drop=1,
    odds_metrics="trimmed_avg",
)
```

## Feature engineering (src/features)

Przykład tworzenia domniemanych prawdopodobienstw z kursow metoda potegowa:

```python
from src.features import add_power_implied_probabilities

df = add_power_implied_probabilities(
    df,
    odds_columns=["max_1", "max_X", "max_2"],
    output_columns=["prob_1", "prob_X", "prob_2"],
    min_odds=1.01,
    initial_k=1.0,
    max_iter=100,
    tolerance=1e-8,
    errors="coerce",
)
```

Działa tez dla rynkow 2-way (np. BTTS):

```python
df = add_power_implied_probabilities(
    df,
    odds_columns=["max_btts_yes", "max_btts_no"],
    output_columns=["prob_btts_yes", "prob_btts_no"],
)
```

Wersja wygodna dla standardowych rynkow (`1x2`, `btts`, `over/under 2.5`)
bez podawania wszystkich kolumn:

```python
from src.features import add_power_implied_probabilities_standard_markets

df = add_power_implied_probabilities_standard_markets(
    df,
    odds_prefix="trimmed_avg",  # domyslnie trimmed_avg
    errors="coerce",
)
```

Mozesz tez wybrac inny prefiks kursow:

```python
df = add_power_implied_probabilities_standard_markets(df, odds_prefix="max")
```

Wyznaczanie bazowych lambd Poissona (`baseline_lambda_home`,
`baseline_lambda_away`) z probabilities 1X2 + over 2.5:

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

Przyklad pelnego pipeline:

```python
from src.features import (
    add_power_implied_probabilities_standard_markets,
    add_baseline_poisson_lambdas,
)

df = (
    df
    .pipe(
        add_power_implied_probabilities_standard_markets,
        odds_prefix="trimmed_avg",
        output_prefix="prob_trimmed_avg",
    )
    .pipe(
        add_baseline_poisson_lambdas,
        prob_home_col="prob_trimmed_avg_1",
        prob_away_col="prob_trimmed_avg_2",
        prob_over25_col="prob_trimmed_avg_over_25",
    )
)
```

## Modelowanie i ewaluacja (src/models)

Warstwa `src/models` zawiera:

- modele statystyczne i ML o wspolnym, prostym API,
- osobny moduł ewaluacji predykcji wynikow (niezalezny od konkretnego modelu).

Aktualnie dostepne:

- `PredictiveModel`, `TrainablePredictiveModel` (`src/models/interfaces`) - wspólne kontrakty dla modeli.
- `PoissonDixonColesModel` (`src/models/statistical`) - predykcja scoreline z korekta Dixon-Colesa,
- `XGBoostPoissonModel` (`src/models/ml`) - para regresorów XGBoost Poisson (home/away) + optymalizacja scoreline,
- `ScoreRule`, `score_single_prediction`, `compute_points_per_match`, `evaluate_score_predictions` (`src/models/evaluation`) - uniwersalna punktacja i metryki.
- `plot_predictions_summary`, `summarize_predictions_1x2`, `PointsSummary1x2` (`src/models/evaluation`) - wizualizacja ewaluacji (rozkład punktów, macierz 1x2).

Przykład uzycia:

```python
from src.models import PoissonDixonColesModel, evaluate_score_predictions

model = PoissonDixonColesModel(
    prob_home_col="prob_1",
    prob_away_col="prob_2",
    prob_over25_col="prob_over_25",
    rho=-0.06,
    bias_correction=1.05,
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

Wizualizacja predykcji (rozkład punktów, macierz 1x2, suma i średnia punktów):

```python
from src.models import plot_predictions_summary

plot_predictions_summary(pred_df, model_name="Poisson Dixon-Coles")
```

Funkcja sama liczy punkty wewnętrznie; domyślne kolumny to `pred_home_goals`, `pred_away_goals`, `home_score`, `away_score`. Do innych analiz możesz użyć `summarize_predictions_1x2`:

```python
from src.models import summarize_predictions_1x2

summary = summarize_predictions_1x2(pred_df)
print(summary.total_points, summary.mean_points)
# summary.points_distribution, summary.outcome_matrix
```

Przykład uzycia modelu XGBoost Poisson (trenowalny, wymaga `fit` przed `predict`):

```python
import xgboost as xgb
from src.models import XGBoostPoissonModel, evaluate_score_predictions

features = [
    "baseline_lambda_home", "baseline_lambda_away",
    "prob_trimmed_avg_1", "prob_trimmed_avg_X", "prob_trimmed_avg_2",
    "prob_trimmed_avg_over_25", "prob_trimmed_avg_btts_yes",
    "margin_avg",
    "value_1", "value_X", "value_2", "value_over25",
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
)

# fit z eval_df (early stopping) lub bez
model.fit(train_df, eval_df=val_df)
pred_df = model.predict(test_df)

metrics = evaluate_score_predictions(pred_df)
```

Model mozna tez uzyc z sezonowym grid searchem (three-way walk-forward):

```python
from src.models.tuning import (
    make_season_walk_forward_splits,
    run_trainable_grid_search_three_way,
)

def xgb_factory(learning_rate=0.025, rho=0.0):
    return XGBoostPoissonModel(
        features_home=features,
        features_away=features,
        model_home=xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=1000,
            max_depth=3,
            learning_rate=learning_rate,
            early_stopping_rounds=50,
        ),
        model_away=xgb.XGBRegressor(
            objective="count:poisson",
            n_estimators=1000,
            max_depth=3,
            learning_rate=learning_rate,
            early_stopping_rounds=50,
        ),
        rho=rho,
    )

folds = make_season_walk_forward_splits(
    historical_df, season_col="season", seasons_order=historical_seasons,
)
search = run_trainable_grid_search_three_way(
    model_factory=xgb_factory,
    param_grid={
        "learning_rate": [0.02, 0.025, 0.03],
        "rho": [0.0, -0.05, -0.10],
    },
    df=historical_df,
    folds=folds,
    score_key="avg_points",
)
```

Przykład pracy przez kontrakt interfejsu:

```python
from src.models import PredictiveModel, PoissonDixonColesModel

model: PredictiveModel = PoissonDixonColesModel(rho=-0.06)
pred_df = model.predict(df)
```

## Tuning modeli (src/models/tuning)

Mozesz uruchomic uniwersalny grid search dla kazdego modelu zgodnego z `PredictiveModel`.
Wspierane sa:

- ranking po `avg_points` (domyslnie) lub innym `score_key`,
- wlasna metryka przez callback `metric_fn`,
- wykres 1D (jeden parametr) i 2D (dwa parametry),
- cache do JSON miedzy uruchomieniami notebooka.

Przykład grid search + cache:

```python
from src.models import (
    PoissonDixonColesModel,
    build_param_grid,
    run_predictive_grid_search,
    plot_grid_search_1d,
)

def model_factory(**params):
    return PoissonDixonColesModel(**params)

param_grid = build_param_grid(
    {
        "rho": {"start": -0.12, "stop": 0.0, "step": 0.04},
    }
)

search = run_predictive_grid_search(
    model_factory=model_factory,
    param_grid=param_grid,
    df=df,
    score_key="avg_points",
    cache_mode="use",  # off/use/refresh
)

ax = plot_grid_search_1d(search.results_df, param_name="rho")
```

Przykład wykresu 2D:

```python
from src.models import plot_grid_search_2d

search_2d = run_predictive_grid_search(
    model_factory=model_factory,
    param_grid={
        "rho": [-0.10, -0.05, 0.0],
        "bias_correction": [0.95, 1.00, 1.05, 1.10],
    },
    df=df,
    cache_mode="use",
)

ax = plot_grid_search_2d(
    search_2d.results_df,
    x_param="bias_correction",
    y_param="rho",
)
```

Wersja dla modeli trenowanych (`TrainablePredictiveModel`) uzywa walidacji czasowej
walk-forward i wykonuje petle:
`fit(train_fold) -> predict(valid_fold) -> evaluate`.

Przykład trainable grid search (time-series):

```python
from src.models import run_trainable_grid_search

trainable_search = run_trainable_grid_search(
    model_factory=trainable_model_factory,
    param_grid={
        "alpha": [0.1, 0.5, 1.0],
        "l2": [0.0, 0.01, 0.1],
    },
    df=df,
    n_splits=4,
    min_train_size=500,
    valid_size=100,
    score_key="avg_points",
    cache_mode="use",
)

# te same helpery wykresow:
ax = plot_grid_search_2d(
    trainable_search.results_df,
    x_param="alpha",
    y_param="l2",
)
```

### **Dwa tryby walk-forward dla modeli trenowalnych**

- **2-way (wierszowy):** `make_walk_forward_splits` + `run_trainable_grid_search` —
`fit(train_df)` (pierwszy argument; `eval_df` opcjonalnie `None`), potem
`predict(valid_df)` na zbiorze walidacyjnym folda — na nim liczone są metryki.
- **3-way (sezonowy):** `make_season_walk_forward_splits` +
`run_trainable_grid_search_three_way` — `fit(train_df, eval_df=val_df)` na sezonie
walidacyjnym (np. early stopping), metryki wyłącznie na osobnym sezonie eval
(bez wycieku val do końcowej oceny).

Przykład sezonowego grid searcha (po wcześniejszym odcięciu holdoutu z `df`):

```python
from src.models.tuning import (
    make_season_walk_forward_splits,
    run_trainable_grid_search_three_way,
)

historical_df = df[df["season"].isin(historical_seasons)].copy()
folds = make_season_walk_forward_splits(
    historical_df,
    season_col="season",
    seasons_order=historical_seasons,
)
search_3 = run_trainable_grid_search_three_way(
    model_factory=trainable_model_factory,
    param_grid={"alpha": [0.1, 0.5]},
    df=historical_df,
    folds=folds,
    datetime_col="match_date",
    score_key="avg_points",
    cache_mode="off",
)
```
