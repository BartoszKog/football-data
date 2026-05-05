import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo
    import pandas as pd
    import numpy as np
    import xgboost as xgb

    from sklearn.model_selection import train_test_split

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Ustawiamy ziarno losowe dla powtarzalności eksperymentu
    np.random.seed(42)

    import sys
    import os

    from src.data import load_and_add_odds_columns_compact
    from src.features import (
        add_baseline_poisson_lambdas,
        add_power_implied_probabilities_standard_markets,
    )

    from src.models.components.matrix_builders import PoissonMatrixBuilder
    from src.models.components.optimizers import ExpectedPointsOptimizer, ExpectedPointsRule
    from src.models import (
        score_single_prediction,
        evaluate_score_predictions,
        plot_predictions_summary,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Prototyp modelu XGBoost Poisson

    Ten notatnik to **eksperymentalny prototyp** modelu XGBoost Poisson do prognozy dokładnych wyników
    meczów (home/away goals) oraz scoringu 1X2. Wykorzystuje on gotowe komponenty z pakietu (`matrix builder`,
    optimizer punktów, funkcje ewaluacyjne) i służy jako miejsce do testowania konfiguracji i cech.

    Pipeline:
    1. Wczytanie danych i wyprowadzenie domniemanych prawdopodobieństw z rynku.
    2. Inżynieria cech (priory Poissona, marże, value, opcjonalnie cechy formy).
    3. Walk-forward cross-validation po sezonach historycznych + grid search hiperparametrów XGBoost i `rho`.
    4. Trening finalnego modelu na wszystkich sezonach historycznych i test na **holdoucie (sezon `current`)**.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Wczytanie danych
    """)
    return


@app.cell
def _():
    df_raw = load_and_add_odds_columns_compact(odds_metrics=["max", "trimmed_avg"])

    print(f"Wczytano {len(df_raw)} meczów.")
    return (df_raw,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Inżynieria cech
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Domniemane prawdopodobieństwa
    """)
    return


@app.cell
def _(df_raw):
    df = (df_raw
        .pipe(
            add_power_implied_probabilities_standard_markets,
            odds_prefix='trimmed_avg',
            output_prefix='prob_trimmed_avg'
        )
        .pipe(
            add_power_implied_probabilities_standard_markets,
            odds_prefix='max',
            output_prefix='prob_max'
        )
    )

    df.head()
    return (df,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cechy z kursów, lambdy i marże
    """)
    return


@app.cell
def _(df):
    df_lam = add_baseline_poisson_lambdas(
        df,
        prob_home_col="prob_trimmed_avg_1",
        prob_away_col="prob_trimmed_avg_2",
        prob_over25_col="prob_trimmed_avg_over_25",
        bias_correction=1.035,
    )

    # 2. Cechy Marży i Szukania Value
    df_lam['value_1'] = df_lam['prob_max_1'] - df_lam['prob_trimmed_avg_1']
    df_lam['value_X'] = df_lam['prob_max_X'] - df_lam['prob_trimmed_avg_X']
    df_lam['value_2'] = df_lam['prob_max_2'] - df_lam['prob_trimmed_avg_2']
    df_lam['value_over25'] = df_lam['prob_max_over_25'] - df_lam['prob_trimmed_avg_over_25']
    df_lam['margin_avg'] = df_lam['prob_trimmed_avg_1'] + df_lam['prob_trimmed_avg_X'] + df_lam['prob_trimmed_avg_2'] - 1.0
    return (df_lam,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Cechy formy

    Poniższa komórka wylicza cechy formy (rolling po wszystkich występach).
    Kolejna komórka z dopiskiem *SANITY CHECK* jest **opcjonalną diagnostyką**.
    """)
    return


@app.cell
def _(df_lam):
    # --- CECHY OGÓLNEJ FORMY (Ostatnie 3 mecze bez podziału na Dom/Wyjazd) ---

    # 1. Tworzymy tymczasową tabelę ze wszystkimi występami (format długi)
    _home_df = df_lam[['match_date', 'home_team', 'home_score', 'away_score']].rename(
        columns={'home_team': 'team', 'home_score': 'scored', 'away_score': 'conceded'}
    )
    _away_df = df_lam[['match_date', 'away_team', 'away_score', 'home_score']].rename(
        columns={'away_team': 'team', 'away_score': 'scored', 'home_score': 'conceded'}
    )

    _df_long = pd.concat([_home_df, _away_df]).sort_values('match_date')

    # 2. Liczymy ogólną formę kroczącą z przesunięciem (tylko ostatnie 3 mecze, żeby łapać aktualne momentum)
    _df_long['overall_scored_roll3'] = _df_long.groupby('team')['scored'].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )
    _df_long['overall_conceded_roll3'] = _df_long.groupby('team')['conceded'].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).mean()
    )

    # 3. Przypinamy wyliczone cechy z powrotem do gospodarzy w głównym DataFrame
    df_form = df_lam.merge(
        _df_long[['match_date', 'team', 'overall_scored_roll3', 'overall_conceded_roll3']],
        left_on=['match_date', 'home_team'],
        right_on=['match_date', 'team'],
        how='left'
    ).rename(columns={
        'overall_scored_roll3': 'home_overall_scored_roll3',
        'overall_conceded_roll3': 'home_overall_conceded_roll3'
    }).drop(columns=['team'])

    # 4. Przypinamy wyliczone cechy z powrotem do gości w głównym DataFrame
    # WAŻNE: merge na df_form (a nie ponownie na df), żeby nie zgubić kolumn home_overall_*
    df_form = df_form.merge(
        _df_long[['match_date', 'team', 'overall_scored_roll3', 'overall_conceded_roll3']],
        left_on=['match_date', 'away_team'],
        right_on=['match_date', 'team'],
        how='left'
    ).rename(columns={
        'overall_scored_roll3': 'away_overall_scored_roll3',
        'overall_conceded_roll3': 'away_overall_conceded_roll3'
    }).drop(columns=['team'])

    # Upewniamy się, że nie powieliliśmy wierszy i przywracamy sortowanie
    df_form = df_form.drop_duplicates(subset=['match_date', 'home_team', 'away_team']).sort_values('match_date')
    return (df_form,)


@app.cell
def _(df_form):
    # --- SANITY CHECK: czy cechy formy są policzone sensownie? ---
    form_cols = [
        "home_overall_scored_roll3",
        "home_overall_conceded_roll3",
        "away_overall_scored_roll3",
        "away_overall_conceded_roll3",
    ]

    missing = [c for c in form_cols if c not in df_form.columns]
    if missing:
        raise ValueError(f"Brak kolumn formy w df_form: {missing}")

    # 1) Duplikaty w kluczu merge (match_date, team) w formacie długim
    _home_df = df_form[["match_date", "home_team", "home_score", "away_score"]].rename(
        columns={"home_team": "team", "home_score": "scored", "away_score": "conceded"}
    )
    _away_df = df_form[["match_date", "away_team", "away_score", "home_score"]].rename(
        columns={"away_team": "team", "away_score": "scored", "home_score": "conceded"}
    )
    _df_long = pd.concat([_home_df, _away_df])
    dup_keys = int(_df_long.duplicated(subset=["match_date", "team"]).sum())

    print("SANITY CHECK - cechy formy")
    print(f"- Duplikaty klucza (match_date, team) w _df_long: {dup_keys}")
    print("- Statystyki cech formy (powinny wyglądać jak średnie goli, zwykle ~0–3):")
    display_stats = df_form[form_cols].describe().T[["count", "mean", "min", "max"]]

    # 2) Przykład dla 1 drużyny (ręczna weryfikacja shift/rolling)
    team = pd.Series(pd.concat([df_form["home_team"], df_form["away_team"]])).dropna().astype(str).iloc[0]
    _team_long = (
        _df_long[_df_long["team"].astype(str) == str(team)]
        .sort_values("match_date")
        .assign(
            scored_shift1=lambda d: d["scored"].shift(1),
            scored_roll3_manual=lambda d: d["scored"].shift(1).rolling(3, min_periods=1).mean(),
        )
        .head(10)
    )

    print(f"- Przykład (pierwsze 10 występów) dla drużyny: {team}")
    _team_long
    return


@app.cell
def _(df_form):
    # Dobór cech: edytuj listę poniżej (np. usuń kolumnę, dodaj nową).
    # features = [
    #     'prob_trimmed_avg_1', 
    #     'prob_trimmed_avg_X', 
    #     'prob_trimmed_avg_2',
    #     'prob_trimmed_avg_over_25', 
    #     'prob_trimmed_avg_under_25',
    #     'prob_trimmed_avg_btts_yes', 
    #     'prob_trimmed_avg_btts_no',
    #     'prob_max_1', 
    #     'prob_max_X', 
    #     'prob_max_2',
    #     'prob_max_over_25', 
    #     'prob_max_under_25',
    #     'prob_max_btts_yes', 
    #     'prob_max_btts_no',
    #     'trimmed_avg_1', 
    #     'trimmed_avg_X', 
    #     'trimmed_avg_2',
    #     'max_1', 
    #     'max_X', 
    #     'max_2',
    #     # --- NOWE CECHY ---
    #     'baseline_lambda_home', 
    #     'baseline_lambda_away',
    #     'value_1', 
    #     'value_X', 
    #     'value_2', 
    #     'value_over25', 
    #     'margin_avg'
    # ]
    features = [
            # 1. Dedykowane priorsy
            'baseline_lambda_home', 
            'baseline_lambda_away',

            # 2. Baza z rynku (Mądrość tłumu)
            'prob_trimmed_avg_1', 
            'prob_trimmed_avg_X', 
            'prob_trimmed_avg_2',
            'prob_trimmed_avg_over_25', 
            'prob_trimmed_avg_btts_yes',

            # 3. Szukanie okazji (Value / Różnice względem MAX)
            'value_1', 
            'value_X', 
            'value_2', 
            # 'value_over25',

            # # --- CECHY FORMY ---
            # 'home_overall_scored_roll3',
            # 'home_overall_conceded_roll3',
            # 'away_overall_scored_roll3',
            # 'away_overall_conceded_roll3'
        ]

    # Pozbywamy się meczów bez kursów lub bez wyniku
    df_clean = df_form.dropna(subset=features + ['home_score', 'away_score']).copy()

    # Oddzielamy X (cechy) i y (cele) dla gospodarzy i gości
    X = df_clean[features]
    y_home = df_clean['home_score'].astype(int)
    y_away = df_clean['away_score'].astype(int)

    print(f"Dane gotowe do treningu: {len(df_clean)} meczów.")
    return X, df_clean, features, y_away, y_home


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Przygotowanie danych do treningu
    """)
    return


@app.cell
def _(X, df_clean, y_away, y_home):
    # --- ZAMKNIĘCIE ZBIORU TESTOWEGO (HOLDOUT) ---
    # Sezon 'current' jest całkowicie wyłączony z pętli walidacyjnej (brak wycieku danych).

    start_day = pd.Timestamp("2025-08-04").tz_localize("Europe/Warsaw")
    mask_holdout = (df_clean["season"] == "current") & (df_clean["match_date"] >= start_day)
    X_holdout = X[mask_holdout]
    y_home_holdout = y_home[mask_holdout]
    y_away_holdout = y_away[mask_holdout]

    # Sezony historyczne do Walk-Forward (bez użycia holdout).
    historical_seasons = [
        "2020/2021",
        "2021/2022",
        "2022/2023",
        "2023/2024",
        "2024/2025",
    ]

    print("Holdout (sezon 'current') zapisany do: X_holdout, y_home_holdout, y_away_holdout (nieużywany w walidacji).")
    print(f"Sezony historyczne do Walk-Forward: {historical_seasons}")
    print(f"Mecze holdout: {mask_holdout.sum()}")
    return X_holdout, historical_seasons, y_away_holdout, y_home_holdout


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Trening modeli XGBoost

    W tej sekcji uruchamiamy **walk-forward cross-validation po sezonach historycznych** oraz prosty
    grid search po wybranych hiperparametrach XGBoost (`learning_rate`, `min_child_weight`, `gamma`)
    i parametrze zależności bramek `rho` w macierzy Poissona. Zakresy siatek są oparte na wcześniejszych
    eksperymentach i eksplorują wąskie okolice dotychczas najlepiej działających ustawień.
    """)
    return


@app.function(hide_code=True)
def run_single_fold(
    X_train,
    y_home_train,
    y_away_train,
    X_val_es,
    y_home_val_es,
    y_away_val_es,
    X_eval,
    df_hist,
    xgb_params,
    builder_params,
    optimizer_params,
):
    """Jeden fold Walk-Forward: trening, (train+val_es) i ewaluacja.

    Zwraca (avg_points, df_eval_fold, model_home, model_away).
    """
    builder = PoissonMatrixBuilder(**builder_params)
    optimizer = ExpectedPointsOptimizer(
        rules=ExpectedPointsRule(), **optimizer_params
    )
    model_home = xgb.XGBRegressor(**xgb_params)
    model_away = xgb.XGBRegressor(**xgb_params)

    # Early stopping opiera się na zbiorze walidacyjnym (val_es),
    # a finalne metryki na osobnym zbiorze eval.
    model_home.fit(
        X_train, y_home_train, eval_set=[(X_val_es, y_home_val_es)], verbose=False
    )
    model_away.fit(
        X_train, y_away_train, eval_set=[(X_val_es, y_away_val_es)], verbose=False
    )

    lambdas_home = model_home.predict(X_eval)
    lambdas_away = model_away.predict(X_eval)

    predictions = []
    for i in range(len(X_eval)):
        l_h, l_a = float(lambdas_home[i]), float(lambdas_away[i])
        matrix = builder.build_matrix(l_h, l_a)
        pred_h, pred_a, _xpts = optimizer.optimize(matrix)
        predictions.append({
            "pred_home_goals": pred_h,
            "pred_away_goals": pred_a,
            "pred_xpts": _xpts,
            "exp_goals_home": l_h,
            "exp_goals_away": l_a,
        })

    pred_df = pd.DataFrame(predictions, index=X_eval.index)
    df_eval_fold = df_hist.loc[pred_df.index].copy()
    df_eval_fold["pred_home_goals"] = pred_df["pred_home_goals"]
    df_eval_fold["pred_away_goals"] = pred_df["pred_away_goals"]
    df_eval_fold["pred_score"] = (
        df_eval_fold["pred_home_goals"].astype(str)
        + ":"
        + df_eval_fold["pred_away_goals"].astype(str)
    )
    df_eval_fold["pred_xpts"] = pred_df["pred_xpts"]
    df_eval_fold["exp_goals_home"] = pred_df["exp_goals_home"]
    df_eval_fold["exp_goals_away"] = pred_df["exp_goals_away"]
    df_eval_fold["points_score"] = df_eval_fold.apply(
        lambda row: score_single_prediction(
            pred_home=row["pred_home_goals"],
            pred_away=row["pred_away_goals"],
            actual_home=row["home_score"],
            actual_away=row["away_score"],
        ),
        axis=1,
    )

    metrics = evaluate_score_predictions(df_eval_fold)
    return metrics["avg_points"], df_eval_fold, model_home, model_away


@app.cell
def _(X, df_clean, historical_seasons, y_away, y_home):
    import itertools

    # --- BAZOWA KONFIGURACJA (stałe parametry; grid search modyfikuje tylko wybrane pola) ---
    BASE_CONFIG = {
        "xgb": {
            "objective": "count:poisson",
            "n_estimators": 1000,
            "max_depth": 3,              # Zablokowane na zwycięskim poziomie, oszczędza czas
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "eval_metric": "poisson-nloglik",
            "early_stopping_rounds": 50,
            "random_state": 42,          # Spójne ziarno losowe dla wszystkich modeli XGB
        },
        "builder": {"max_goals_matrix": 6},
        "optimizer": {"max_goals_prediction": 4, "max_goals_matrix": 6},
    }

    # Siatka hiperparametrów XGBoost (eksploracja wokół wcześniej sprawdzonych, dobrych wartości)
    param_grid_xgb = {
        "learning_rate": [0.02, 0.025, 0.03],
        "min_child_weight": [4, 5, 6],          # Wąski zakres wokół poprzednio najlepszego = 5
        "gamma": [0.1, 0.15, 0.2],
    }

    # Siatka hiperparametrów Buildera (rho; przesuwamy w stronę zera vs wcześniejsze ustawienia)
    param_grid_builder = {
        "rho": [0.00, -0.05, -0.10],            # Przesuwamy w stronę zera!
    }

    xgb_param_names = list(param_grid_xgb.keys())
    xgb_grid_values = [param_grid_xgb[name] for name in xgb_param_names]
    rho_values = param_grid_builder["rho"]

    # Wszystkie kombinacje: (parametry XGB, rho)
    combos = list(
        itertools.product(
            *xgb_grid_values,
            rho_values,
        )
    )
    total_combos = len(combos)

    # --- Przygotowanie danych historycznych (bez holdout 'current') ---
    mask_historical = df_clean["season"].isin(historical_seasons)
    X_hist = X[mask_historical]
    y_home_hist = y_home[mask_historical]
    y_away_hist = y_away[mask_historical]
    df_hist = df_clean[mask_historical]

    best_score = float("-inf")
    best_xgb_params = None
    best_builder_params = None
    best_df_eval = None
    best_df_eval_folds = None
    best_model_home = None
    best_model_away = None

    # --- Iteracja po kombinacjach hiperparametrów (XGBoost + rho) ---
    for combo_idx, combo in enumerate(combos, start=1):
        # Rozpakuj kombinację na parametry XGBoost + rho
        *xgb_values, rho_value = combo

        # Budujemy słownik parametrów XGBoost dla tej kombinacji
        xgb_params = BASE_CONFIG["xgb"].copy()
        for name, value in zip(xgb_param_names, xgb_values):
            xgb_params[name] = value

        # Budujemy słownik parametrów buildera (rho + stałe)
        builder_params = BASE_CONFIG["builder"].copy()
        builder_params["rho"] = rho_value

        print(
            f"\n=== Kombinacja {combo_idx}/{total_combos}: "
            f"max_depth={xgb_params['max_depth']}, "
            f"learning_rate={xgb_params['learning_rate']}, "
            f"min_child_weight={xgb_params['min_child_weight']}, "
            f"gamma={xgb_params['gamma']}, "
            f"rho={builder_params['rho']} ==="
        )

        avg_points_per_fold = []
        df_eval_folds = []

        # --- Walk-Forward: Eval = sezon N, Val (early stopping) = N-1, Train = sezony 0..N-2 ---
        for eval_idx in range(2, len(historical_seasons)):
            train_seasons = historical_seasons[: eval_idx - 1]
            val_season = historical_seasons[eval_idx - 1]
            eval_season = historical_seasons[eval_idx]

            mask_train = df_hist["season"].isin(train_seasons)
            mask_val_es = df_hist["season"] == val_season
            mask_eval = df_hist["season"] == eval_season

            X_train = X_hist[mask_train]
            X_val_es = X_hist[mask_val_es]
            X_eval = X_hist[mask_eval]
            y_home_train = y_home_hist[mask_train]
            y_away_train = y_away_hist[mask_train]
            y_home_val_es = y_home_hist[mask_val_es]
            y_away_val_es = y_away_hist[mask_val_es]

            fold_num = eval_idx - 1
            print(
                f"Fold {fold_num}: "
                f"Train={train_seasons}, Val={val_season}, Eval={eval_season}"
            )

            print("  Trenowanie...")
            avg_pts, df_eval_fold, model_home, model_away = run_single_fold(
                X_train,
                y_home_train,
                y_away_train,
                X_val_es,
                y_home_val_es,
                y_away_val_es,
                X_eval,
                df_hist,
                xgb_params,
                builder_params,
                BASE_CONFIG["optimizer"],
            )

            avg_points_per_fold.append(avg_pts)
            df_eval_folds.append(df_eval_fold)
            print(f"  Średnia pkt na mecz (fold): {avg_pts:.4f}")

        # --- Średni wynik CV dla danej kombinacji ---
        mean_cv = (
            sum(avg_points_per_fold) / len(avg_points_per_fold)
            if avg_points_per_fold
            else float("nan")
        )
        print(
            f"--> Średnia CV (avg_points) dla kombinacji "
            f"{combo_idx}/{total_combos}: {mean_cv:.4f}"
        )

        # --- Aktualizacja najlepszej kombinacji w całym grid searchu ---
        if mean_cv > best_score:
            best_score = mean_cv
            best_xgb_params = xgb_params.copy()
            best_builder_params = builder_params.copy()
            best_df_eval_folds = df_eval_folds
            best_df_eval = df_eval_folds[-1] if df_eval_folds else None
            best_model_home = model_home
            best_model_away = model_away

    print("\n=== NAJLEPSZA KOMBINACJA ===")
    if best_xgb_params is not None and best_builder_params is not None:
        print(f"BEST SCORE (avg_points, CV): {best_score:.4f}")
        print("BEST XGB PARAMS:")
        for k, v in best_xgb_params.items():
            print(f"  {k}: {v}")
        print("BEST BUILDER PARAMS:")
        for k, v in best_builder_params.items():
            print(f"  {k}: {v}")
    else:
        print("Brak udanej kombinacji hiperparametrów (sprawdź dane / sezony).")

    # Dla dalszych komórek: ostatni fold i modele z najlepszej kombinacji
    df_eval = best_df_eval
    df_eval_folds = best_df_eval_folds
    model_home = best_model_home
    model_away = best_model_away
    return (
        best_builder_params,
        best_xgb_params,
        df_eval,
        df_eval_folds,
        model_away,
        model_home,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Ewaluacja

    Poniższe komórki pokazują wyniki z **ostatniego folda walk-forward** (tabela mecz po meczu,
    metryki globalne oraz wykresy diagnostyczne), żeby można było szybko ocenić zachowanie modelu
    na aktualnym sezonie walidacyjnym.
    """)
    return


@app.cell
def _(df_eval):
    # df_eval z ostatniego kroku Walk-Forward (do wizualizacji)
    _columns_to_show = [
        "match_date", "home_team", "away_team", "trimmed_avg_1", "trimmed_avg_X", "trimmed_avg_2",
        "home_score", "away_score", "pred_score", "points_score", "pred_xpts", "exp_goals_home", "exp_goals_away",
    ]
    print("Tabela ewaluacji z ostatniego folda Walk-Forward.")
    df_eval[_columns_to_show]
    return


@app.cell
def _(df_eval):
    evaluate_score_predictions(df=df_eval)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Wykresy diagnostyczne
    """)
    return


@app.cell
def _(df_eval):
    plot_predictions_summary(df_eval, model_name="Ostatni fold - rozkład punktów i 1X2")
    return


@app.cell
def _(df_eval_folds):
    plot_predictions_summary(
        df_eval_folds[1], model_name="Fold 2 - rozkład punktów i 1X2"
    )
    return


@app.cell
def _(df_eval_folds):
    plot_predictions_summary(
        df_eval_folds[0], model_name="Fold 1 - rozkład punktów i 1X2"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Ważność cech
    """)
    return


@app.cell
def _(model_home):
    xgb.plot_importance(
        model_home, 
        importance_type='gain'
    )
    return


@app.cell
def _(model_away):
    xgb.plot_importance(
        model_away, 
        importance_type='gain'
    )
    return


@app.cell
def _(model_away, model_home):
    # Wyciągamy słowniki z wartościami 'gain' bezpośrednio z silnika (Booster)
    gain_home = model_home.get_booster().get_score(importance_type='gain')
    gain_away = model_away.get_booster().get_score(importance_type='gain')

    # Zamieniamy na czytelne DataFrames
    df_gain_home = pd.DataFrame(list(gain_home.items()), columns=['Cecha', 'Gain_Gospodarze'])
    df_gain_away = pd.DataFrame(list(gain_away.items()), columns=['Cecha', 'Gain_Goście'])

    # Łączymy obie tabele w jedną po nazwie cechy
    df_importance = pd.merge(df_gain_home, df_gain_away, on='Cecha', how='outer').fillna(0)

    # Dodajemy kolumnę z sumą, żeby posortować globalnie od najważniejszych
    df_importance['Gain_Suma'] = df_importance['Gain_Gospodarze'] + df_importance['Gain_Goście']
    df_importance = df_importance.sort_values(by='Gain_Suma', ascending=False).reset_index(drop=True)

    # Usuwamy kolumnę pomocniczą i formatujemy liczby dla czytelności
    df_importance = df_importance.drop(columns=['Gain_Suma'])
    df_importance['Gain_Gospodarze'] = df_importance['Gain_Gospodarze'].round(4)
    df_importance['Gain_Goście'] = df_importance['Gain_Goście'].round(4)

    # Wyświetlamy całą tabelę
    df_importance
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Test ostateczny

    Na końcu trenujemy **ostateczny model** na wszystkich sezonach historycznych z użyciem
    najlepszych hiperparametrów z walk-forward grid search i testujemy go na niezależnym
    zbiorze **holdout (sezon `current`)**, który nie był użyty w procesie strojenia.
    """)
    return


@app.cell
def _(
    X_holdout,
    best_builder_params,
    best_xgb_params,
    df_clean,
    features,
    historical_seasons,
    y_away_holdout,
    y_home_holdout,
):
    # --- OSTATECZNY TEST NA ZBIORZE HOLDOUT (SEZON 'current') ---

    print("=== TRENOWANIE OSTATECZNEGO MODELU ===")
    # Trenujemy na WSZYSTKICH danych historycznych
    X_train_final = df_clean[df_clean["season"].isin(historical_seasons)][features]
    y_home_final = df_clean[df_clean["season"].isin(historical_seasons)]["home_score"]
    y_away_final = df_clean[df_clean["season"].isin(historical_seasons)]["away_score"]

    # Do Early Stopping użyjemy ostatniego dostępnego sezonu historycznego (np. 2024/2025)
    mask_val_final = df_clean["season"] == historical_seasons[-1]
    X_val_final = df_clean[mask_val_final][features]
    y_home_val_final = df_clean[mask_val_final]["home_score"]
    y_away_val_final = df_clean[mask_val_final]["away_score"]

    # Inicjalizacja modeli z NAJLEPSZYMI parametrami z walk-forward grid search
    final_model_home = xgb.XGBRegressor(**best_xgb_params)
    final_model_away = xgb.XGBRegressor(**best_xgb_params)

    final_model_home.fit(X_train_final, y_home_final, eval_set=[(X_val_final, y_home_val_final)], verbose=False)
    final_model_away.fit(X_train_final, y_away_final, eval_set=[(X_val_final, y_away_val_final)], verbose=False)

    print("=== PREDYKCJA NA SEZONIE 'CURRENT' ===")
    lambdas_home_holdout = final_model_home.predict(X_holdout)
    lambdas_away_holdout = final_model_away.predict(X_holdout)

    # Builder z najlepszym znalezionym rho (i stałym max_goals_matrix z grid searchu)
    final_builder = PoissonMatrixBuilder(**best_builder_params)
    final_optimizer = ExpectedPointsOptimizer(rules=ExpectedPointsRule(), max_goals_prediction=4, max_goals_matrix=6)

    holdout_predictions = []
    for i in range(len(X_holdout)):
        l_h, l_a = float(lambdas_home_holdout[i]), float(lambdas_away_holdout[i])
        matrix = final_builder.build_matrix(l_h, l_a)
        pred_h, pred_a, _xpts = final_optimizer.optimize(matrix)

        holdout_predictions.append({
            "pred_home_goals": pred_h,
            "pred_away_goals": pred_a,
        })

    df_holdout_eval = X_holdout.copy()
    df_holdout_eval["pred_home_goals"] = [p["pred_home_goals"] for p in holdout_predictions]
    df_holdout_eval["pred_away_goals"] = [p["pred_away_goals"] for p in holdout_predictions]
    df_holdout_eval["home_score"] = y_home_holdout
    df_holdout_eval["away_score"] = y_away_holdout

    # Używamy Twojej funkcji ewaluacyjnej z scoring.py
    final_metrics = evaluate_score_predictions(df_holdout_eval)

    print(f"\n🏆 WYNIK NA SEZONIE CURRENT 🏆")
    print(f"Rozegrane mecze: {final_metrics['matches_evaluated']}")
    print(f"Zdobyte punkty: {final_metrics['total_points']}")
    print(f"Średnia pkt/mecz: {final_metrics['avg_points']:.4f}")
    return (df_holdout_eval,)


@app.cell
def _(df_holdout_eval):
    df_holdout_eval
    return


@app.cell
def _(df_holdout_eval):
    plot_predictions_summary(df_holdout_eval, model_name="Zbiór testowy - rozkład punktów i 1X2")
    return


if __name__ == "__main__":
    app.run()
