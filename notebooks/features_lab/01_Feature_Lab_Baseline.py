import marimo

__generated_with = "0.22.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import numpy as np
    import pandas as pd
    import xgboost as xgb
    import shap

    from sklearn.inspection import permutation_importance

    import matplotlib.pyplot as plt

    import scipy.stats

    import json
    import os

    from src.data import load_and_add_odds_columns_compact
    from src.features import (
        add_baseline_poisson_lambdas,
        add_power_implied_probabilities_standard_markets,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Konfiguracja i Wczytanie Danych
    """)
    return


@app.cell
def _():
    df_raw = load_and_add_odds_columns_compact(odds_metrics="trimmed_avg")
    return (df_raw,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Tworzenie Cech (Definicje i Aplikacja)
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.1 Definiowanie funkcji transformujących
    """)
    return


@app.cell
def _():
    # Pipeline functions are imported from src.features (Cell 1).
    # Define additional custom feature-engineering functions here.
    #
    # Convention: new features that differ between home and away should use
    # the _home / _away suffix so that make_long_format picks them up
    # automatically.
    #
    # def add_my_new_feature(df: pd.DataFrame) -> pd.DataFrame:
    #     """Add new feature columns (with _home/_away suffix convention)."""
    #     ...
    #     return df
    pass
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.2 Aplikacja pipelinów (Wide Format)
    """)
    return


@app.cell
def _(df_raw):
    df_wide_full = (
        df_raw.pipe(
            add_power_implied_probabilities_standard_markets,
            odds_prefix="trimmed_avg",
            output_prefix="prob_trimmed_avg",
        ).pipe(
            add_baseline_poisson_lambdas,
            prob_home_col="prob_trimmed_avg_1",
            prob_away_col="prob_trimmed_avg_2",
            prob_over25_col="prob_trimmed_avg_over_25",
            bias_correction=1.035,
        )
        # .pipe(add_my_new_feature)
    )
    print(
        f"Wide format — {len(df_wide_full)} meczów, "
        f"{len(df_wide_full.columns)} kolumn"
    )
    return (df_wide_full,)


@app.cell
def _(df_wide_full):
    df_wide_full
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## 2.3 Transformacja Wide → Long
    """)
    return


@app.function
def make_long_format(
    df: pd.DataFrame,
    home_suffix: str = "_home",
    away_suffix: str = "_away",
) -> pd.DataFrame:
    """Wide (1 row/match) -> Long (2 rows/match).

    Columns ending with *home_suffix* whose counterpart (*away_suffix*)
    also exists are treated as paired features and renamed to
    ``team_<base>`` / ``opponent_<base>``.  All other columns are kept
    as shared match-level features.  An ``is_home`` flag and a
    ``match_id`` (original index) column are added.
    """
    bases = []
    for col in df.columns:
        if col.endswith(home_suffix):
            base = col[: -len(home_suffix)]
            if base + away_suffix in df.columns:
                bases.append(base)

    paired: set[str] = set()
    for b in bases:
        paired.add(b + home_suffix)
        paired.add(b + away_suffix)
    shared_cols = [c for c in df.columns if c not in paired]

    home_data = df[shared_cols].copy()
    for b in bases:
        home_data[f"team_{b}"] = df[b + home_suffix].values
        home_data[f"opponent_{b}"] = df[b + away_suffix].values
    home_data["is_home"] = 1
    home_data["match_id"] = df.index.to_numpy()

    away_data = df[shared_cols].copy()
    for b in bases:
        away_data[f"team_{b}"] = df[b + away_suffix].values
        away_data[f"opponent_{b}"] = df[b + home_suffix].values
    away_data["is_home"] = 0
    away_data["match_id"] = df.index.to_numpy()

    return pd.concat([home_data, away_data], ignore_index=True)


@app.cell
def _(df_wide_full):
    df_long_full = make_long_format(df_wide_full)
    df_long_full["team_score"] = np.where(
        df_long_full["is_home"] == 1,
        df_long_full["home_score"],
        df_long_full["away_score"],
    )
    print(
        f"Long format — {len(df_long_full)} wierszy "
        f"({len(df_long_full) // 2} meczów)"
    )
    return (df_long_full,)


@app.cell
def _(df_long_full):
    df_long_full
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Panel Sterowania Eksperymentem
    """)
    return


@app.cell
def _(df_long_full, df_wide_full):
    _long_feature_candidates = sorted([
        c
        for c in df_long_full.columns
        if (c.startswith(("team_", "opponent_", "prob_")) or c == "is_home")
        and c != "team_score"
    ])
    _wide_feature_candidates = sorted([
        c
        for c in df_wide_full.columns
        if c.startswith(("prob_", "baseline_lambda_"))
    ])

    _long_defaults = [
        c
        for c in ["team_baseline_lambda", "opponent_baseline_lambda"]
        if c in _long_feature_candidates
    ]
    _wide_defaults = [
        c
        for c in ["baseline_lambda_home", "baseline_lambda_away"]
        if c in _wide_feature_candidates
    ]

    use_long_model = mo.ui.switch(label="Jeden model (format długi)")
    model_type = mo.ui.dropdown(
        options=["xgb", "dummy"],
        value="xgb",
        label="Typ modelu",
    )
    feature_selector_long = mo.ui.multiselect(
        options=_long_feature_candidates,
        value=_long_defaults,
        label="Cechy (Long)",
    )
    feature_selector_home = mo.ui.multiselect(
        options=_wide_feature_candidates,
        value=_wide_defaults,
        label="Cechy (Home)",
    )
    feature_selector_away = mo.ui.multiselect(
        options=_wide_feature_candidates,
        value=_wide_defaults,
        label="Cechy (Away)",
    )
    return (
        feature_selector_away,
        feature_selector_home,
        feature_selector_long,
        model_type,
        use_long_model,
    )


@app.cell
def _(
    feature_selector_away,
    feature_selector_home,
    feature_selector_long,
    model_type,
    use_long_model,
):
    _row_long = mo.hstack(
        [
            feature_selector_long,
            mo.md(f"Selected: `{list(feature_selector_long.value)}`"),
        ],
        align="start",
    )
    _row_home = mo.hstack(
        [
            feature_selector_home,
            mo.md(f"Selected: `{list(feature_selector_home.value)}`"),
        ],
        align="start",
    )
    _row_away = mo.hstack(
        [
            feature_selector_away,
            mo.md(f"Selected: `{list(feature_selector_away.value)}`"),
        ],
        align="start",
    )

    mo.vstack([
        use_long_model,
        model_type,
        mo.md("---"),
        mo.md("**Cechy — tryb Long (jeden model)**"),
        _row_long,
        mo.md("**Cechy — tryb Wide (dwa modele)**"),
        _row_home,
        _row_away,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. Modele Dummy i Podział Danych
    """)
    return


@app.class_definition
class DummyLambdaModel:
    """Predict-only baseline that returns a single lambda column verbatim."""

    def __init__(self, lambda_col_name: str):
        self.lambda_col_name = lambda_col_name

    def fit(self, X, y):
        pass

    def predict(self, X):
        return X[self.lambda_col_name].to_numpy()


@app.cell
def _(
    df_long_full,
    df_wide_full,
    feature_selector_away,
    feature_selector_home,
    feature_selector_long,
    use_long_model,
):
    _seasons_train = ["2020/2021", "2021/2022", "2022/2023"]
    _season_es = "2023/2024"
    _season_val = "2024/2025"

    if use_long_model.value:
        _features = list(feature_selector_long.value)
        _target = "team_score"
        mo.stop(
            not _features,
            mo.md("**Wybierz co najmniej jedną cechę (Long).**"),
        )
        _df = df_long_full.dropna(subset=_features + [_target]).copy()
        _train_m = _df["season"].isin(_seasons_train)
        _es_m = _df["season"] == _season_es
        _val_m = _df["season"] == _season_val

        bundle = {
            "mode": "long_single",
            "X_train": _df.loc[_train_m, _features],
            "X_es": _df.loc[_es_m, _features],
            "X_val": _df.loc[_val_m, _features],
            "y_train": _df.loc[_train_m, _target],
            "y_es": _df.loc[_es_m, _target],
            "y_val": _df.loc[_val_m, _target],
            "val_long_meta": _df.loc[
                _val_m, ["match_id", "is_home", _target]
            ].rename(columns={_target: "score"}),
        }

        _text = (
            f"[LONG] Trening: **{len(bundle['X_train'])}** wierszy | "
            f"ES: **{len(bundle['X_es'])}** | "
            f"Walidacja: **{len(bundle['X_val'])}**"
        )
    else:
        _features_home = list(feature_selector_home.value)
        _features_away = list(feature_selector_away.value)
        _target_home = "home_score"
        _target_away = "away_score"
        mo.stop(
            not _features_home or not _features_away,
            mo.md("**Wybierz co najmniej jedną cechę (Home i Away).**"),
        )
        _feature_union = sorted(set(_features_home) | set(_features_away))
        _df = df_wide_full.dropna(
            subset=_feature_union + [_target_home, _target_away]
        ).copy()
        _train_m = _df["season"].isin(_seasons_train)
        _es_m = _df["season"] == _season_es
        _val_m = _df["season"] == _season_val

        bundle = {
            "mode": "two_models",
            "X_train_home": _df.loc[_train_m, _features_home],
            "X_train_away": _df.loc[_train_m, _features_away],
            "X_es_home": _df.loc[_es_m, _features_home],
            "X_es_away": _df.loc[_es_m, _features_away],
            "X_val_home": _df.loc[_val_m, _features_home],
            "X_val_away": _df.loc[_val_m, _features_away],
            "y_home_train": _df.loc[_train_m, _target_home],
            "y_away_train": _df.loc[_train_m, _target_away],
            "y_home_es": _df.loc[_es_m, _target_home],
            "y_away_es": _df.loc[_es_m, _target_away],
            "y_home_val": _df.loc[_val_m, _target_home],
            "y_away_val": _df.loc[_val_m, _target_away],
        }

        _text = (
            f"[DWA MODELE] Trening: **{len(bundle['X_train_home'])}** meczów | "
            f"ES: **{len(bundle['X_es_home'])}** | "
            f"Walidacja: **{len(bundle['X_val_home'])}**"
        )    

    _info = mo.md(_text)
    _info
    return (bundle,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 5. Trening
    """)
    return


@app.cell
def _(bundle, model_type):
    _lab_params = {
        "objective": "count:poisson",
        "eval_metric": "poisson-nloglik",
        "n_estimators": 2000,
        "early_stopping_rounds": 50,
        "max_depth": 4,
        "learning_rate": 0.05,
        "min_child_weight": 3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
    }

    model_home = None
    model_away = None
    model_long = None

    if model_type.value == "xgb":
        if bundle["mode"] == "two_models":
            model_home = xgb.XGBRegressor(**_lab_params)
            model_away = xgb.XGBRegressor(**_lab_params)
            model_home.fit(
                bundle["X_train_home"],
                bundle["y_home_train"],
                eval_set=[(bundle["X_es_home"], bundle["y_home_es"])],
                verbose=True,
            )
            model_away.fit(
                bundle["X_train_away"],
                bundle["y_away_train"],
                eval_set=[(bundle["X_es_away"], bundle["y_away_es"])],
                verbose=True,
            )
        else:
            model_long = xgb.XGBRegressor(**_lab_params)
            model_long.fit(
                bundle["X_train"],
                bundle["y_train"],
                eval_set=[(bundle["X_es"], bundle["y_es"])],
                verbose=True,
            )
    else:
        if bundle["mode"] == "two_models":
            mo.stop(
                "baseline_lambda_home" not in bundle["X_train_home"].columns
                or "baseline_lambda_away"
                not in bundle["X_train_away"].columns,
                mo.md(
                    "**Dummy wymaga kolumn `baseline_lambda_home` / "
                    "`baseline_lambda_away` w wybranych cechach.**"
                ),
            )
            model_home = DummyLambdaModel("baseline_lambda_home")
            model_away = DummyLambdaModel("baseline_lambda_away")
            model_home.fit(bundle["X_train_home"], bundle["y_home_train"])
            model_away.fit(bundle["X_train_away"], bundle["y_away_train"])
        else:
            mo.stop(
                "team_baseline_lambda" not in bundle["X_train"].columns,
                mo.md(
                    "**Dummy wymaga kolumny `team_baseline_lambda` "
                    "w wybranych cechach.**"
                ),
            )
            model_long = DummyLambdaModel("team_baseline_lambda")
            model_long.fit(bundle["X_train"], bundle["y_train"])
    return model_away, model_home, model_long


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 6. Ewaluacja, SHAP
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Błędy modeli (wspólna ramka walidacyjna)
    """)
    return


@app.function
def evaluate_predictions(y_true_home, y_pred_home, y_true_away, y_pred_away):
    """Compute per-match Poisson deviance for home and away predictions.

    Returns means, standard errors of the mean (``SE = std(..., ddof=1) / sqrt(N)``),
    and ``Error_Vector``: Python list ``[dev_home for each match, then dev_away for each match]``.
    """

    def _poisson_deviance_per_sample(y_true, y_pred) -> np.ndarray:
        """Per-observation Poisson deviance (same aggregation as sklearn mean_poisson_deviance)."""
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64).ravel(), 1e-15)

        # Oszukujemy logarytm: tam gdzie y_true to 0, sztucznie podmieniamy na 1.0 (tylko do logarytmu)
        safe_y_true = np.where(y_true > 0, y_true, 1.0)

        # Teraz liczymy term1 używając safe_y_true. 
        # NumPy nie zobaczy log(0), a np.where i tak wyzeruje te miejsca.
        term1 = np.where(y_true > 0, y_true * np.log(safe_y_true / y_pred), 0.0)
        term2 = y_true - y_pred
        return 2.0 * (term1 - term2)

    dev_home = _poisson_deviance_per_sample(y_true_home, y_pred_home)
    dev_away = _poisson_deviance_per_sample(y_true_away, y_pred_away)
    n = int(dev_home.shape[0])
    if n == 0:
        raise ValueError("evaluate_predictions requires at least one match.")
    se_home = float(np.std(dev_home, ddof=1) / np.sqrt(n))
    se_away = float(np.std(dev_away, ddof=1) / np.sqrt(n))
    per_match_mean = (dev_home + dev_away) / 2.0
    se_mean = float(np.std(per_match_mean, ddof=1) / np.sqrt(n))
    err_vec = np.concatenate([dev_home, dev_away])
    return {
        "Deviance_home": round(float(np.mean(dev_home)), 4),
        "SE_home": round(se_home, 4),
        "Deviance_away": round(float(np.mean(dev_away)), 4),
        "SE_away": round(se_away, 4),
        "Deviance_mean": round(float(np.mean(per_match_mean)), 4),
        "SE_mean": round(se_mean, 4),
        "Error_Vector": err_vec.tolist(),
    }


@app.function
def compare_models_statistically(current_vector, best_vector, alpha=0.05):
    """Paired t-test on per-observation deviances (current vs best).

    Lower mean deviance is better. Returns ``comparison_status`` without emojis:
    ``better_significant``, ``better_not_significant``, ``worse``, or ``error``.
    """
    a = np.asarray(current_vector, dtype=float).ravel()
    b = np.asarray(best_vector, dtype=float).ravel()
    if a.shape != b.shape or a.size == 0:
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "alpha": alpha,
            "comparison_status": "error",
            "mean_current": float("nan"),
            "mean_best": float("nan"),
            "message": "Niezgodna długość wektorów lub pusty wektor.",
        }
    mean_c = float(np.mean(a))
    mean_b = float(np.mean(b))
    better = mean_c < mean_b
    tt = scipy.stats.ttest_rel(a, b)
    pval = float(tt.pvalue) if tt.pvalue is not None else float("nan")
    stat = float(tt.statistic)

    if not better:
        return {
            "statistic": stat,
            "pvalue": pval,
            "alpha": alpha,
            "comparison_status": "worse",
            "mean_current": mean_c,
            "mean_best": mean_b,
            "message": (
                "Bieżący model nie ma niższej średniej odchyleniowej Poissona "
                "niż najlepszy zapis w historii."
            ),
        }
    if pval < alpha:
        return {
            "statistic": stat,
            "pvalue": pval,
            "alpha": alpha,
            "comparison_status": "better_significant",
            "mean_current": mean_c,
            "mean_best": mean_b,
            "message": (
                "Bieżący model jest istotnie lepszy (p < α) od najlepszego zapisu w historii."
            ),
        }
    return {
        "statistic": stat,
        "pvalue": pval,
        "alpha": alpha,
        "comparison_status": "better_not_significant",
        "mean_current": mean_c,
        "mean_best": mean_b,
        "message": (
            "Średnia odchyleniowa jest niższa niż u najlepszego zapisu, "
            "ale różnica nie jest istotna statystycznie przy tym teście."
        ),
    }


@app.cell
def _(bundle, model_away, model_home, model_long):
    if bundle["mode"] == "two_models":
        _y_pred_home = model_home.predict(bundle["X_val_home"])
        _y_pred_away = model_away.predict(bundle["X_val_away"])
        eval_metrics = evaluate_predictions(
            bundle["y_home_val"].to_numpy(),
            _y_pred_home,
            bundle["y_away_val"].to_numpy(),
            _y_pred_away,
        )
    else:
        _meta = bundle["val_long_meta"].copy()
        _meta["pred"] = model_long.predict(bundle["X_val"])
        _h = _meta[_meta["is_home"] == 1][["match_id", "score", "pred"]].rename(
            columns={"score": "y_home", "pred": "pred_home"}
        )
        _a = _meta[_meta["is_home"] == 0][["match_id", "score", "pred"]].rename(
            columns={"score": "y_away", "pred": "pred_away"}
        )
        _eval_df = _h.merge(_a, on="match_id", how="inner")
        eval_metrics = evaluate_predictions(
            _eval_df["y_home"].to_numpy(),
            _eval_df["pred_home"].to_numpy(),
            _eval_df["y_away"].to_numpy(),
            _eval_df["pred_away"].to_numpy(),
        )

    mo.output.append(mo.md("---"))

    _pair_rows = [
        ("Deviance_home", "SE_home"),
        ("Deviance_away", "SE_away"),
        ("Deviance_mean", "SE_mean"),
    ]
    _rows = [
        "| Metryka | Wartość (średnia ± SE) |",
        "|---------|------------------------|",
    ]
    for _dk, _sk in _pair_rows:
        _rows.append(
            f"| {_dk} | **{eval_metrics[_dk]} ± {eval_metrics[_sk]}** |"
        )

    _text = "\n".join(_rows)
    mo.output.append(mo.md(_text))
    return (eval_metrics,)


@app.function
def compute_permutation_importance_table(model, X, y, features, model_label):
    """Compute and print permutation importance for a fitted model."""
    _perm_results = permutation_importance(
        model,
        X,
        y,
        scoring="neg_mean_poisson_deviance",
        n_repeats=10,
        random_state=42,
        n_jobs=-1,
    )

    _perm_df = pd.DataFrame(
        {
            "Cecha": features,
            "Ważność (spadek błędu)": _perm_results.importances_mean,
            "Odchylenie (std)": _perm_results.importances_std,
        }
    ).sort_values(by="Ważność (spadek błędu)", ascending=False)

    mo.output.append(mo.md(f"Wyniki Tasowania (Permutation Importance) - {model_label}:"))
    mo.output.append(mo.md(_perm_df.to_markdown(index=False)))
    return


@app.cell
def _(bundle, model_away, model_home, model_long, model_type):
    home_shap_values = None
    away_shap_values = None
    long_shap_values = None

    if model_type.value == "xgb":
        if bundle["mode"] == "two_models":
            _home_explainer = shap.TreeExplainer(model_home)
            home_shap_values = _home_explainer(bundle["X_val_home"])
            _away_explainer = shap.TreeExplainer(model_away)
            away_shap_values = _away_explainer(bundle["X_val_away"])
        else:
            _long_explainer = shap.TreeExplainer(model_long)
            long_shap_values = _long_explainer(bundle["X_val"])
    return away_shap_values, home_shap_values, long_shap_values


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Gospodarze (tylko tryb dwóch modeli)
    """)
    return


@app.cell
def _(bundle, feature_selector_home, model_home, model_type):
    if bundle["mode"] == "two_models" and model_home is not None and model_type.value == "xgb":
        compute_permutation_importance_table(
            model=model_home,
            X=bundle["X_val_home"],
            y=bundle["y_home_val"],
            features=list(feature_selector_home.value),
            model_label="GOSPODARZE",
        )
    return


@app.cell
def _(bundle, home_shap_values):
    if bundle["mode"] == "two_models" and home_shap_values is not None:
        mo.output.append(mo.md("**GOSPODARZE**"))
        _ax = shap.plots.beeswarm(home_shap_values, show=False)
        mo.output.append(_ax)
    return


@app.cell
def _(bundle, home_shap_values):
    if bundle["mode"] == "two_models" and home_shap_values is not None:
        mo.output.append(mo.md("**GOSPODARZE**"))
        shap.plots.scatter(home_shap_values, show=False)
        _fig = plt.gcf()
        mo.output.append(_fig)
        plt.close(_fig)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Goście (tylko tryb dwóch modeli)
    """)
    return


@app.cell
def _(bundle, feature_selector_away, model_away, model_type):
    if bundle["mode"] == "two_models" and model_away is not None and model_type.value == "xgb":
        compute_permutation_importance_table(
            model=model_away,
            X=bundle["X_val_away"],
            y=bundle["y_away_val"],
            features=list(feature_selector_away.value),
            model_label="GOŚCIE",
        )
    return


@app.cell
def _(away_shap_values, bundle):
    if bundle["mode"] == "two_models" and away_shap_values is not None:
        mo.output.append(mo.md("**GOŚCIE**"))
        _ax = shap.plots.beeswarm(away_shap_values, show=False)
        mo.output.append(_ax)
    return


@app.cell
def _(away_shap_values, bundle):
    if bundle["mode"] == "two_models" and away_shap_values is not None:
        mo.output.append(mo.md("**GOŚCIE**"))
        shap.plots.scatter(away_shap_values, show=False)
        _fig = plt.gcf()
        mo.output.append(_fig)
        plt.close(_fig)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Jeden model — Permutation (format długi)
    """)
    return


@app.cell
def _(bundle, feature_selector_long, model_long, model_type):
    if bundle["mode"] == "long_single" and model_long is not None and model_type.value == "xgb":
        compute_permutation_importance_table(
            model=model_long,
            X=bundle["X_val"],
            y=bundle["y_val"],
            features=list(feature_selector_long.value),
            model_label="LONG (jeden model)",
        )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Jeden model — SHAP (format długi)
    """)
    return


@app.cell
def _(bundle, long_shap_values):
    if bundle["mode"] == "long_single" and long_shap_values is not None:
        mo.output.append(mo.md("**LONG (jeden model)**"))
        _ax = shap.plots.beeswarm(long_shap_values, show=False)
        mo.output.append(_ax)
    return


@app.cell
def _(bundle, long_shap_values):
    if bundle["mode"] == "long_single" and long_shap_values is not None:
        plt.figure()
        _available_features = long_shap_values.feature_names
        _color_param = (
            long_shap_values[:, "is_home"]
            if "is_home" in _available_features
            else None
        )
        shap.plots.scatter(long_shap_values, color=_color_param, show=False)
        _fig = plt.gcf()
        mo.output.append(_fig)
        plt.close(_fig)
    return


@app.cell
def _(eval_metrics):
    _path = os.path.join("logs", "feature_experiments_01_log.csv")

    mo.stop(
        not os.path.exists(_path),
        mo.md("Brak historii do porównania statystycznego.")
    )

    _hist = pd.read_csv(_path)
    _dm = pd.to_numeric(_hist["Deviance_mean"], errors="coerce")
    _best_idx = _dm.idxmin()
    _best_row = _hist.loc[_best_idx]
    _ev_raw = _best_row["Error_Vector"]

    try:
        _best_vec = json.loads(_ev_raw)
    except (json.JSONDecodeError, TypeError):
        _best_vec = None

    mo.stop(
        _best_vec is None,
        mo.md("**Niepoprawny format Error_Vector w historii.**")
    )

    _result = compare_models_statistically(
        eval_metrics["Error_Vector"],
        _best_vec,
        alpha=0.05,
    )

    _status = _result.get("comparison_status", "error")

    mo.stop(
        _status == "error",
        mo.md(f"**{_result.get('message', 'Błąd porównania')}**")
    )

    _emoji_map = {
        "better_significant": "🟢",
        "better_not_significant": "🟠",
        "worse": "🔴",
    }
    _emoji = _emoji_map.get(_status, "—")

    _p = _result.get("pvalue")
    if _p is not None and not (isinstance(_p, float) and np.isnan(_p)):
        _p_str = f"{_p:.4g}"
    else:
        _p_str = "n/a"

    mo.md(
        f"### Porównanie statystyczne z najlepszym zapisem w historii\n\n"
        f"**{_emoji}** `comparison_status={_status}`\n\n"
        f"- **p-wartość** (test t dla par, `scipy.stats.ttest_rel`): **{_p_str}**\n"
        f"- **Średnia bieżąca / średnia najlepszego zapisu**: "
        f"{_result.get('mean_current', float('nan')):.6g} / "
        f"{_result.get('mean_best', float('nan')):.6g}\n\n"
        f"{_result.get('message', '')}"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Dziennik Decyzji (Notatki)
    """)
    return


@app.cell
def _():
    get_save_clicks, set_save_clicks = mo.state(0)
    return get_save_clicks, set_save_clicks


@app.cell
def _():
    experiment_notes = mo.ui.text_area(label="📝 Notatki do tego eksperymentu:")
    save_button = mo.ui.button(
        value=0,
        on_click=lambda value: value + 1,
        label="💾 Zapisz wynik do dziennika",
    )
    mo.vstack([experiment_notes, save_button])
    return experiment_notes, save_button


@app.cell(hide_code=True)
def _(
    eval_metrics,
    experiment_notes,
    feature_selector_away,
    feature_selector_home,
    feature_selector_long,
    get_save_clicks,
    model_type,
    save_button,
    set_save_clicks,
    use_long_model,
):
    _log_file = os.path.join("logs", "feature_experiments_01_log.csv")
    os.makedirs(os.path.dirname(_log_file), exist_ok=True)

    if save_button.value is not None and save_button.value > get_save_clicks():
        set_save_clicks(save_button.value)

        _notes = experiment_notes.value
        if _notes is not None and str(_notes).strip() != "":
            _metrics_row = {
                k: v
                for k, v in eval_metrics.items()
                if k != "Error_Vector"
            }
            _metrics_row["Error_Vector"] = json.dumps(
                eval_metrics["Error_Vector"],
                ensure_ascii=False,
            )
            _new_data = {
                "Data": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Typ_Modelu": model_type.value,
                "Tryb_Danych": (
                    "long_single" if use_long_model.value else "two_models"
                ),
                **_metrics_row,
                "Cechy_JSON": json.dumps(
                    {
                        "tryb_danych": (
                            "long_single"
                            if use_long_model.value
                            else "two_models"
                        ),
                        "long": list(feature_selector_long.value),
                        "wide_home": list(feature_selector_home.value),
                        "wide_away": list(feature_selector_away.value),
                    },
                    ensure_ascii=False,
                ),
                "Notatki": _notes,
            }
            _new_entry = pd.DataFrame([_new_data])

            _should_save = True
            if os.path.exists(_log_file):
                _existing_df = pd.read_csv(_log_file)
                _cols_to_compare = [
                    c
                    for c in _new_entry.columns
                    if c not in ("Data", "Error_Vector") and c in _existing_df.columns
                ]
                if (
                    _cols_to_compare
                    and len(_existing_df) > 0
                    and _existing_df[_cols_to_compare]
                    .iloc[-1:]
                    .reset_index(drop=True)
                    .equals(
                        _new_entry[_cols_to_compare].reset_index(drop=True)
                    )
                ):
                    _should_save = False

            if _should_save:
                if os.path.exists(_log_file):
                    _combined = pd.concat(
                        [_existing_df, _new_entry], ignore_index=True
                    )
                    _combined.to_csv(_log_file, index=False)
                else:
                    _new_entry.to_csv(_log_file, index=False)

    if os.path.exists(_log_file):
        _df_history = pd.read_csv(_log_file)
        _df_display = _df_history.drop(columns=["Error_Vector"], errors="ignore")
        _output = mo.ui.table(_df_display, selection=None)
    else:
        _output = mo.md("*Brak historii eksperymentów. Zapisz pierwszy wynik!*")

    _output
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
