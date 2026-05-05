import marimo

__generated_with = "0.23.5"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.models.components import PoissonMatrixBuilder, average_scoreline_nll
    from src.models.evaluation import pearson_chi2_scoreline_gof


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Pearson χ² Diagnostics Lab (Synthetic Data)

    Notebook pokazuje praktyczne użycie `pearson_chi2_scoreline_gof`:
    - jak działa agregacja 4x4 (`0, 1, 2, 3+`) i merge do `Other`,
    - jak zachowuje się diagnostyka przy sweepie po `rho`,
    - jak łączyć tę diagnostykę z NLL (NLL jako objective, Pearson jako kontrola dopasowania).
    """)
    return


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return (is_script_mode,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 1) Ustawienia eksperymentu

    Tu definiujesz dwa zestawy parametrów:
    - **generator danych**: `n_matches`, `seed`, `rho_true`,
    - **diagnostyka/sweep**: zakres `rho`, `min_expected_threshold`, `ddof`.

    Cel: stworzyć kontrolowane dane i sprawdzić, jak stabilnie Pearson χ²
    reaguje na zmianę `rho`.
    """)
    return


@app.cell
def _():
    experiment_form = (
        mo.md(
            """
            {n_matches}
            {random_seed}
            {rho_true}
            """
        )
        .batch(
            n_matches=mo.ui.slider(
                start=400,
                stop=6000,
                step=200,
                value=2000,
                show_value=True,
                label="Liczba symulowanych meczów",
            ),
            random_seed=mo.ui.number(value=20260505, label="Seed"),
            rho_true=mo.ui.slider(
                start=-0.20,
                stop=0.20,
                step=0.01,
                value=-0.08,
                show_value=True,
                label="Prawdziwe rho (generator danych)",
            ),
        )
        .form(submit_button_label="Zastosuj ustawienia generatora")
    )
    experiment_form
    return (experiment_form,)


@app.cell
def _():
    sweep_form = (
        mo.md(
            """
            {rho_start}
            {rho_stop}
            {rho_step}
            {min_expected_threshold}
            {ddof}
            {max_goals_matrix}
            """
        )
        .batch(
            rho_start=mo.ui.slider(
                start=-0.25,
                stop=0.20,
                step=0.01,
                value=-0.20,
                show_value=True,
                label="Rho sweep: start",
            ),
            rho_stop=mo.ui.slider(
                start=-0.20,
                stop=0.25,
                step=0.01,
                value=0.10,
                show_value=True,
                label="Rho sweep: stop",
            ),
            rho_step=mo.ui.slider(
                start=0.01,
                stop=0.10,
                step=0.01,
                value=0.02,
                show_value=True,
                label="Rho sweep: step",
            ),
            min_expected_threshold=mo.ui.slider(
                start=1.0,
                stop=20.0,
                step=1.0,
                value=5.0,
                show_value=True,
                label="Pearson: min expected threshold",
            ),
            ddof=mo.ui.number(value=0, start=0, step=1, label="Pearson: ddof"),
            max_goals_matrix=mo.ui.slider(
                start=4,
                stop=12,
                step=1,
                value=8,
                show_value=True,
                label="max_goals_matrix",
            ),
        )
        .form(submit_button_label="Zastosuj ustawienia sweep/diagnostyki")
    )
    sweep_form
    return (sweep_form,)


@app.cell
def _(experiment_form, sweep_form):
    experiment_values = experiment_form.value or {
        "n_matches": 2000,
        "random_seed": 20260505,
        "rho_true": -0.08,
    }
    sweep_values = sweep_form.value or {
        "rho_start": -0.20,
        "rho_stop": 0.10,
        "rho_step": 0.02,
        "min_expected_threshold": 5.0,
        "ddof": 0,
        "max_goals_matrix": 8,
    }
    n_matches = int(experiment_values["n_matches"])
    random_seed = int(experiment_values["random_seed"])
    rho_true = float(experiment_values["rho_true"])

    rho_start = float(sweep_values["rho_start"])
    rho_stop = float(sweep_values["rho_stop"])
    rho_step = float(sweep_values["rho_step"])
    min_expected_threshold = float(sweep_values["min_expected_threshold"])
    ddof = int(sweep_values["ddof"])
    max_goals_matrix = int(sweep_values["max_goals_matrix"])
    return (
        ddof,
        max_goals_matrix,
        min_expected_threshold,
        n_matches,
        random_seed,
        rho_start,
        rho_step,
        rho_stop,
        rho_true,
    )


@app.cell
def _(max_goals_matrix, n_matches, random_seed, rho_true):
    rng = np.random.default_rng(random_seed)
    n_obs = n_matches

    # Heterogeniczne intensywności meczu (bardziej realistyczne niż stałe lambdy).
    tempo = rng.normal(loc=0.0, scale=0.22, size=n_obs)
    home_strength = rng.normal(loc=0.0, scale=0.30, size=n_obs)
    away_strength = rng.normal(loc=0.0, scale=0.30, size=n_obs)
    lambda_home = np.exp(0.25 + tempo + home_strength)
    lambda_away = np.exp(-0.03 + tempo + away_strength)

    true_builder = PoissonMatrixBuilder(
        rho=rho_true,
        max_goals_matrix=max_goals_matrix,
    )

    sampled_home = np.empty(n_obs, dtype=np.intp)
    sampled_away = np.empty(n_obs, dtype=np.intp)
    for idx, (lam_h, lam_a) in enumerate(zip(lambda_home, lambda_away, strict=True)):
        matrix = true_builder.build_matrix(float(lam_h), float(lam_a))
        chosen = int(rng.choice(matrix.size, p=matrix.ravel()))
        sampled_home[idx], sampled_away[idx] = np.unravel_index(chosen, matrix.shape)

    synthetic_df = pd.DataFrame(
        {
            "lambda_home": lambda_home,
            "lambda_away": lambda_away,
            "home_score": sampled_home,
            "away_score": sampled_away,
        }
    )

    mo.md(f"""
          Wygenerowano {len(synthetic_df)} meczów.

          Średnie gole:
          - gospodarzy: `{synthetic_df["home_score"].mean():.2f}`
          - gości: `{synthetic_df["away_score"].mean():.2f}`
    """)
    return lambda_away, lambda_home, sampled_away, sampled_home


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 2) Generacja danych syntetycznych

    Każdy mecz dostaje własne `lambda_home` i `lambda_away` (tempo + siła),
    a wynik jest losowany z macierzy scoreline z `rho_true`.

    To ważne: wynik obserwowany jest próbkowany z dokładnie tego samego typu
    macierzy (`PoissonMatrixBuilder`), którego później używamy w ewaluacji.
    Jedyna różnica w sweepie to wartość `rho`.

    Czyli porównujemy:
    - **data-generating process**: macierz Dixon-Coles z `rho_true`,
    - **modele oceniane**: ta sama konstrukcja macierzy, ale z `rho_candidate`.

    Dzięki temu testujesz wpływ samego parametru `rho` na dopasowanie, bez
    mieszania różnych klas modeli.
    """)
    return


@app.cell
def _(rho_start, rho_step, rho_stop):
    start = rho_start
    stop = rho_stop
    step = rho_step
    if stop < start:
        start, stop = stop, start
    rho_grid = np.arange(start, stop + step * 0.5, step, dtype=np.float64)
    if rho_grid.size == 0:
        rho_grid = np.array([0.0], dtype=np.float64)
    grid_info = mo.md(
        f"**Rho sweep:** `{rho_grid[0]:.2f}` .. `{rho_grid[-1]:.2f}` "
        f"(krok `{step:.2f}`, {rho_grid.size} punktów)"
    )
    return grid_info, rho_grid


@app.cell
def _(grid_info):
    grid_info
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 3) Sweep po `rho`

    Dla każdego kandydata `rho` budujemy macierz prawdopodobieństw *tym samym*
    `PoissonMatrixBuilder`, którego użyliśmy do generacji danych, ale z inną
    wartością `rho`.

    Następnie liczymy:
    - `avg_nll` (metryka celu),
    - `pearson_chi2_scoreline_gof` (diagnostyka kalibracji),
    - wskaźniki pomocnicze: `chi2/dof`, `pvalue`, liczba scalonych binów.

    To rozdziela **optymalizację** (NLL) od **diagnostyki kalibracji** (χ²).
    """)
    return


@app.cell
def _(
    ddof,
    lambda_away,
    lambda_home,
    max_goals_matrix,
    min_expected_threshold,
    rho_grid,
    sampled_away,
    sampled_home,
):
    rows: list[dict[str, float | int]] = []
    bins_by_rho: dict[float, pd.DataFrame] = {}

    for rho_candidate in rho_grid:
        builder = PoissonMatrixBuilder(
            rho=float(rho_candidate),
            max_goals_matrix=max_goals_matrix,
        )
        avg_nll, n_used = average_scoreline_nll(
            lambda_home,
            lambda_away,
            sampled_home,
            sampled_away,
            rho=float(rho_candidate),
            max_goals_matrix=max_goals_matrix,
        )
        pearson = pearson_chi2_scoreline_gof(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
            actual_home=sampled_home,
            actual_away=sampled_away,
            matrix_builder=builder,
            min_expected_threshold=min_expected_threshold,
            ddof=ddof,
        )
        chi2_per_dof = (
            float(pearson.chi2_stat / pearson.dof)
            if pearson.dof > 0 and np.isfinite(pearson.chi2_stat)
            else np.nan
        )
        rows.append(
            {
                "rho": float(rho_candidate),
                "avg_nll": float(avg_nll),
                "chi2_stat": float(pearson.chi2_stat),
                "dof": int(pearson.dof),
                "chi2_per_dof": float(chi2_per_dof) if np.isfinite(chi2_per_dof) else np.nan,
                "pvalue": float(pearson.pvalue),
                "n_bins_merged": int(pearson.n_bins_merged),
                "n_bins_after_merge": int(pearson.n_bins_after_merge),
                "n_matches_used": int(n_used),
            }
        )
        bins_by_rho[round(float(rho_candidate), 8)] = pearson.bins_df.copy()

    results_df = pd.DataFrame(rows).sort_values("avg_nll", ascending=True).reset_index(drop=True)
    return bins_by_rho, results_df


@app.cell
def _(results_df):
    mo.md(f"""
    **Policzono {len(results_df)} kandydatów `rho`.**
    Najlepszy NLL: `{results_df.loc[0, 'avg_nll']:.5f}` dla `rho={results_df.loc[0, 'rho']:.3f}`.
    """)
    return


@app.cell
def _(results_df):
    results_df
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 4) Podgląd binów i sanity checks

    `bins_df` pokazuje wkład każdego bina 4x4 (+ ewentualnie `Other`) do χ².
    To tutaj zobaczysz, które segmenty rozkładu najbardziej psują dopasowanie.

    Sanity checks kontrolują:
    - zgodność sum `observed` i `expected`,
    - liczbę binów po mergowaniu,
    - sensowność stopni swobody.
    """)
    return


@app.cell
def _(rho_true):
    selected_rho = mo.ui.number(value=rho_true, step=0.01, label="Podgląd bins_df dla rho")
    selected_rho
    return (selected_rho,)


@app.cell
def _(bins_by_rho: dict[float, pd.DataFrame], selected_rho):
    key = min(
        bins_by_rho.keys(),
        key=lambda value: abs(value - float(selected_rho.value)),
    )
    bins_preview = bins_by_rho[key].copy()
    bins_preview = bins_preview.sort_values(["is_other", "bin"], ascending=[True, True]).reset_index(drop=True)
    mo.md(f"Najbliższe dostępne `rho`: `{key:.3f}`")
    return bins_preview, key


@app.cell
def _(bins_preview):
    bins_preview
    return


@app.cell
def _(bins_preview, ddof, key):
    observed_sum = float(bins_preview["observed"].sum())
    expected_sum = float(bins_preview["expected"].sum())
    n_after = int((bins_preview["used_in_test"] == True).sum())
    dof_inferred = n_after - 1 - ddof
    sanity_df = pd.DataFrame(
        {
            "check": [
                "sum(expected) ~= sum(observed)",
                "n_bins_after_merge <= 17",
                "pvalue ma sens gdy dof > 0",
            ],
            "status": [
                bool(np.isclose(expected_sum, observed_sum, atol=1e-6)),
                bool(n_after <= 17),
                bool(dof_inferred > 0),
            ],
            "detail": [
                f"expected={expected_sum:.6f}, observed={observed_sum:.0f}",
                f"n_bins_after_merge={n_after}",
                f"dof_inferred={dof_inferred}",
            ],
        }
    )
    mo.md(f"Sanity checks dla `rho={key:.3f}`")
    return (sanity_df,)


@app.cell
def _(sanity_df):
    sanity_df
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## 5) Jak czytać wykresy

    - `rho vs avg_nll`: minimum wskazuje obszar najlepszy predykcyjnie.
    - `rho vs pvalue`: niskie wartości sugerują słabą zgodność agregatów.
    - `rho vs chi2/dof`: okolice `1.0` zwykle wyglądają zdrowiej diagnostycznie.

    Praktycznie: wybieraj `rho` najpierw po NLL, a χ² traktuj jako filtr jakości
    kalibracji (plus końcowe potwierdzenie w PIT).
    """)
    return


@app.cell
def _(results_df):
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))

    plot_rho_df = results_df.sort_values(by="rho")

    axes[0].plot(plot_rho_df["rho"], plot_rho_df["avg_nll"], marker="o", linewidth=1.2)
    axes[0].set_title("Rho vs avg_nll")
    axes[0].set_xlabel("rho")
    axes[0].set_ylabel("avg_nll")
    axes[0].grid(alpha=0.3)

    axes[1].plot(plot_rho_df["rho"], plot_rho_df["pvalue"], marker="o", linewidth=1.2, color="C2")
    axes[1].axhline(0.05, linestyle="--", linewidth=1.0, color="C3")
    axes[1].set_title("Rho vs p-value")
    axes[1].set_xlabel("rho")
    axes[1].set_ylabel("p-value")
    axes[1].grid(alpha=0.3)

    axes[2].plot(plot_rho_df["rho"], plot_rho_df["chi2_per_dof"], marker="o", linewidth=1.2, color="C1")
    axes[2].axhline(1.0, linestyle="--", linewidth=1.0, color="C3")
    axes[2].set_title("Rho vs chi2/dof")
    axes[2].set_xlabel("rho")
    axes[2].set_ylabel("chi2/dof")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Kiedy to narzędzie jest przydatne

    - **NLL**: używaj do strojenia (`objective`) — to główny sygnał jakości predykcyjnej.
    - **Pearson χ² (4x4 + Other)**: używaj do sprawdzania, czy zagregowany rozkład wyników
        nie jest systematycznie rozjechany względem obserwacji.
    - **PIT**: używaj jako uzupełnienie do oglądu kształtu kalibracji i lokalnych odchyleń.

    Praktyczny workflow:
    1. sweep po `rho` i ranking po NLL,
    2. odrzuć kandydatów z bardzo słabą zgodnością χ² (`chi2/dof` wysoko, niskie `pvalue`),
    3. finalnie potwierdź wybór diagnostyką PIT.
    """)
    return


@app.cell(hide_code=True)
def _(is_script_mode):
    mode_text = "script" if is_script_mode else "interactive"
    mo.md(f"Tryb uruchomienia marimo: **{mode_text}**")
    return


if __name__ == "__main__":
    app.run()
