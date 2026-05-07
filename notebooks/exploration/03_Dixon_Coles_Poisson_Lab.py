import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    from src.data import load_and_add_odds_columns_compact
    from src.features import (
        add_calibrated_poisson_lambdas,
        add_power_implied_probabilities_standard_markets,
    )
    from src.models import (
        evaluate_poisson_deviance,
        plot_predictions_summary,
        plot_predictions_scoreline_summary,
        compute_points_per_match,
    )
    from src.models.components import (
        PoissonMatrixBuilder,
        ExpectedPointsOptimizer,
        ExpectedPointsRule,
        calibrate_rho,
        plot_rho_calibration,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Dane — skalibrowane lambdy Poissona

    Wczytanie danych z kursów, obliczenie implied probabilities
    i skalibrowanych lambd według **`exp(intercept + slope · λ_base)`**
    (tożsame z `exp(slope · λ_base + intercept)`).

    Domyślne **`intercept = −0.354611`**, **`slope = 0.443665`** są tymi samymi
    wartościami co w `add_calibrated_poisson_lambdas` w pakiecie — pochodzą z
    `notebooks/exploration/02_GAM_Lab.py`: tam kalibracja GLM (łącze logowe,
    efektywnie λ skalowane tą funkcją) została dobrana tak, że na zbiorze
    walidacyjnym **dewiacja Poissona** była niższa niż dla λ z modelu XGBoost
    (przy najlepszym zestawie ustawień sprawdzonym w tamtym laboratorium).
    """)
    return


@app.cell
def _():
    df_raw = load_and_add_odds_columns_compact(odds_metrics="trimmed_avg")
    df = (
        df_raw.pipe(
            add_power_implied_probabilities_standard_markets,
            output_prefix = "prob_trimmed_avg"
        ).pipe(
            add_calibrated_poisson_lambdas,
        )
    )
    mo.md(
        f"**Załadowano {len(df)} meczów, {len(df.columns)} kolumn.**\n\n"
        f"Kalibracja (jak wyżej — źródło: `notebooks/exploration/02_GAM_Lab.py`): "
        f"`intercept={-0.354611}`, `slope={0.443665}`"
    )
    return (df,)


@app.cell
def _(df):
    mo.ui.dataframe(
        df[
            [
                "season",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "calibrated_lambda_home",
                "calibrated_lambda_away",
            ]
        ]
    )
    return


@app.cell
def _(df):
    np.unique(df['season'])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Ewaluacja — Poisson Deviance (sezon 2024/25)
    """)
    return


@app.cell
def _(df):
    SEASON_VAL = "2024/2025"

    df_val = df.loc[
        df["season"] == SEASON_VAL,
        [
            "home_score",
            "away_score",
            "calibrated_lambda_home",
            "calibrated_lambda_away",
        ],
    ].dropna()

    y_true_home = df_val["home_score"].to_numpy(dtype=np.float64)
    y_true_away = df_val["away_score"].to_numpy(dtype=np.float64)
    pred_home = df_val["calibrated_lambda_home"].to_numpy(dtype=np.float64)
    pred_away = df_val["calibrated_lambda_away"].to_numpy(dtype=np.float64)

    eval_metrics = evaluate_poisson_deviance(
        y_true_home, pred_home, y_true_away, pred_away,
    )

    _pair_rows = [
        ("Deviance_home", "SE_home"),
        ("Deviance_away", "SE_away"),
        ("Deviance_mean", "SE_mean"),
    ]
    _table_lines = [
        "| Metryka | Wartość (średnia ± SE) |",
        "|---------|------------------------|",
    ]
    for _dk, _sk in _pair_rows:
        _table_lines.append(
            f"| {_dk} | **{eval_metrics[_dk]} ± {eval_metrics[_sk]}** |"
        )

    mo.md(
        f"**Sezon walidacyjny:** `{SEASON_VAL}` · "
        f"**N meczów:** {len(df_val)}\n\n"
        + "\n".join(_table_lines)
        + "\n\n"
        + r"*Te wartości pokrywają się z dewiacją Poissona liczoną na tym samym "
        r"zbiorze walidacyjnym w `notebooks/exploration/02_GAM_Lab.py`, "
        r"w którym dostrajano kalibrację GLM (`intercept` / `slope`).*"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Dixon-Coles — optymalizacja ρ

    Grid-search parametru ρ korekty Dixon-Coles (wyniki 0-0, 0-1, 1-0, 1-1)
    na zbiorze walidacyjnym. Kryterium: **average negative log-likelihood**
    prawdziwego wyniku w macierzy Poissona.
    """)
    return


@app.cell
def _(df):
    SEASONS_RHO = ["2020/2021", "2021/2022", "2022/2023", "2023/2024", "2024/2025"]

    df_rho_src = df.loc[
        df["season"].isin(SEASONS_RHO),
        ["home_score", "away_score", "calibrated_lambda_home", "calibrated_lambda_away"],
    ].dropna()

    rho_result = calibrate_rho(
        lambda_home=df_rho_src["calibrated_lambda_home"].to_numpy(),
        lambda_away=df_rho_src["calibrated_lambda_away"].to_numpy(),
        actual_home=df_rho_src["home_score"].to_numpy(dtype=int),
        actual_away=df_rho_src["away_score"].to_numpy(dtype=int),
    )
    return SEASONS_RHO, rho_result


@app.cell
def _(SEASONS_RHO, rho_result):
    _ax = plot_rho_calibration(rho_result)
    _ax.set_title(
        f"Dixon-Coles \u03c1 Grid Search"
        f"  \u00b7  {len(SEASONS_RHO)} sezon\u00f3w, N={rho_result.n_matches}"
    )
    _ax.figure.tight_layout()

    _seasons_str = ", ".join(SEASONS_RHO)
    mo.vstack([
        mo.md(
            f"Optymalny \u03c1 = **{rho_result.best_rho:.2f}**"
            f"  (Avg NLL = {rho_result.best_nll:.5f},"
            f" mecze: {rho_result.n_matches},"
            f" sezony: {_seasons_str})"
        ),
        _ax.figure,
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. Test punktacji na zbiorze walidacyjnym i testowym

    **Cel.** Ocenić jakość **typowanych wyników meczu** (nie samych λ ani ρ): jak często trafiamy
    dokładny wynik, różnicę bramek albo sam obrót 1X2 — przy domyślnej regule punktów z `ScoreRule`.

    **Reguła punktów** (wartości domyślne): dokładny wynik **3 pkt**, poprawna różnica bramek **2 pkt**,
    poprawny wynik meczu (1 / X / 2) **1 pkt**, pudło **0 pkt**. Wykres z `plot_predictions_summary`
    pokazuje rozkład tych punktów oraz macierz pomyłek 1X2 (rzeczywisty × typowany obrót).

    **`pred_score`** to dyskretny typ **h:a** po optymalizacji oczekiwanych punktów na macierzy
    Dixon–Coles (zaokrąglone cele bramkowe z optymalizera), a nie „najbardziej prawdopodobny” wynik
    prosto z rozkładu Poissona.

    **Zbiory.** Walidacja: pełny sezon **2024/2025**. Test: **`current`** od ustalonej daty — dla niektórych
    meczów wynik może być jeszcze niedostępny (wtedy nie wchodzą do porównań wymagających rzeczywistego `h:a`).

    **Niżej:** przy każdym sezonie najpierw tabela i podsumowanie punktacji, potem
    **`plot_predictions_scoreline_summary`** — cztery panele: dwie heatmapy par bramek (ścięcie **≥ 4 → 4**;
    wspólna skala kolorów) oraz pod spodem **top 6** najczęstszych pełnych wyników **h:a** (typ vs rzeczywistość, bez ścięcia).
    Nie pokazujemy marginalnych rozkładów scoreline na jednym wykresie słupkowym (łatwo o mylące „sumy” częstości).
    """)
    return


@app.cell
def _(df, rho_result):
    _builder = PoissonMatrixBuilder(rho=rho_result.best_rho, max_goals_matrix=10)
    _optimizer = ExpectedPointsOptimizer(
        rules=ExpectedPointsRule(),
        max_goals_prediction=4,
        max_goals_matrix=10,
    )

    def _add_predictions(df_slice):
        _preds = []
        for _, _row in df_slice.iterrows():
            _lh = _row["calibrated_lambda_home"]
            _la = _row["calibrated_lambda_away"]
            if pd.isna(_lh) or pd.isna(_la):
                _preds.append((pd.NA, pd.NA, pd.NA))
                continue
            _mat = _builder.build_matrix(float(_lh), float(_la))
            _ph, _pa, _xpts = _optimizer.optimize(_mat)
            _preds.append((_ph, _pa, _xpts, f"{_ph}:{_pa}"))
        _out = df_slice.copy()
        _out[["pred_home_goals", "pred_away_goals", "pred_xpts", "pred_score"]] = _preds
        return _out

    df_pred_2425 = _add_predictions(
        df.loc[df["season"] == "2024/2025"]
        .dropna(subset=["calibrated_lambda_home", "calibrated_lambda_away"])
    )

    start_day = pd.Timestamp("2025-08-04").tz_localize("Europe/Warsaw")
    df_pred_current = _add_predictions(
        df.loc[
            (df["season"] == "current") & (df["match_date"] >= start_day)
        ].dropna(subset=["calibrated_lambda_home", "calibrated_lambda_away"])
    )

    mo.md(
        f"**Predykcje gotowe** \u00b7 "
        f"\u03c1 = {rho_result.best_rho:.2f} \u00b7 "
        f"2024/25: {len(df_pred_2425)} mecz\u00f3w \u00b7 "
        f"current (od {start_day.date()}): {len(df_pred_current)} meczów"
    )
    return df_pred_2425, df_pred_current, start_day


@app.cell
def _():
    colnames_to_show = [
        "match_date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "calibrated_lambda_home",
        "calibrated_lambda_away",
        "pred_home_goals",
        "pred_away_goals",
        "pred_xpts",
        "points_score",
        "real_score",
        "pred_score",
    ]
    return (colnames_to_show,)


@app.cell
def _(df_pred_2425, df_pred_current):
    def format_score(row):
        if pd.isnull(row['home_score']) or pd.isnull(row['away_score']):
            return "N/A"
        return f"{int(row['home_score'])}:{int(row['away_score'])}"

    df_pred_2425['points_score'] = compute_points_per_match(df_pred_2425)
    df_pred_2425['real_score'] = df_pred_2425.apply(format_score, axis=1)

    df_pred_current['points_score'] = compute_points_per_match(df_pred_current)
    df_pred_current['real_score'] = df_pred_current.apply(format_score, axis=1)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sezon 2024/25 - walidacyjny
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tabela wyników
    """)
    return


@app.cell
def _(colnames_to_show, df_pred_2425):
    df_pred_2425[colnames_to_show]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Podsumowanie punktacji
    """)
    return


@app.cell
def _(df_pred_2425, rho_result):
    _fig_2425 = plot_predictions_summary(
        df_pred_2425,
        model_name=f"Dixon-Coles (\u03c1={rho_result.best_rho:.2f}) \u00b7 2024/25",
    )
    mo.output.append(_fig_2425)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Wyniki h:a — `plot_predictions_scoreline_summary`

    Jedna figura **2×2**: **heatmapy** par bramek (ścięcie **≥ 4 → 4**, wspólna skala kolorów) oraz
    **top 6** najczęstszych pełnych wyników **h:a** (bez ścięcia). Uwzględniane są wiersze, w których predykcje
    i faktyczny wynik są liczbowo dostępne — jak w sygnaturze funkcji w pakiecie.
    """)
    return


@app.cell
def _(df_pred_2425, rho_result):
    _fig_sl = plot_predictions_scoreline_summary(
        df_pred_2425,
        model_name=f"Dixon-Coles (\u03c1={rho_result.best_rho:.2f}) \u00b7 2024/25 \u00b7 h:a",
    )
    mo.output.append(_fig_sl)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Sezon 2025/26 - testowy
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Tabela wyników
    """)
    return


@app.cell
def _(colnames_to_show, df_pred_current):
    df_pred_current[colnames_to_show]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Podsumowanie punktacji
    """)
    return


@app.cell
def _(df_pred_current, rho_result, start_day):
    _fig_cur = plot_predictions_summary(
        df_pred_current,
        model_name=f"Dixon-Coles (\u03c1={rho_result.best_rho:.2f}) \u00b7 current (od {start_day.date()})",
    )
    mo.output.append(_fig_cur)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    Punktowanie przy tych skalibrowanych λ i wybranym ρ Dixon–Coles jest tu **bardzo przeciętne**.
    Modele dostrajane wcześniej w projekcie osiągały **zdecydowanie wyższy** poziom punktacji.

    Poniżej — jak przy walidacji — **`plot_predictions_scoreline_summary`** (heatmapy + top 6).
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Wyniki h:a — `plot_predictions_scoreline_summary`

    Jak przy walidacji: heatmapy ze ścięciem **≥ 4 → 4** oraz **top 6** pełnych **h:a**.
    """)
    return


@app.cell
def _(df_pred_current, rho_result, start_day):
    _fig_sl_cur = plot_predictions_scoreline_summary(
        df_pred_current,
        model_name=(
            f"Dixon-Coles (\u03c1={rho_result.best_rho:.2f}) \u00b7 current "
            f"(od {start_day.date()}) \u00b7 h:a"
        )
    )
    mo.output.append(_fig_sl_cur)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Wniosek

    **Optymalizacja λ i wybory oparte o dewiację Poissona** w dotychczasowym ujęciu **nie są dobrym pomysłem**:
    mimo starannego strojenia, **predykcje (w tym punktacja) pozostają przeciętne** w porównaniu z wcześniejszymi modelami w projekcie.

    ## Następne kroki w projekcie

    1. **Optymalizacja modeli pod NLL** (ujemne log-prawdopodobieństwo) zamiast pośrednich heurystyk związanych z dewiacją.
    2. **Weryfikacja kalibracji λ**: **PIT** (probability integral transform) oraz **test chi-kwadrat Pearsona** jako formalne narzędzia oceny zgodności rozkładów z danymi.
    """)
    return


if __name__ == "__main__":
    app.run()
