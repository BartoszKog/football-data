import marimo

__generated_with = "0.23.4"
app = marimo.App(width="medium")

with app.setup:
    from dataclasses import dataclass
    from pathlib import Path
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.models.components import PoissonMatrixBuilder
    from src.models.evaluation.pit import (
        build_pit_diagnostics,
        plot_pit_histogram_replicates,
        plot_pit_worm_replicates,
    )


@app.cell(hide_code=True)
def _():
    mo.md("""
    # PIT Diagnostics Simulation Lab


    Ten notebook jest celowo edukacyjny: generujemy dane z kontrolowanego,
    znanego procesu, a potem oceniamy kilka modeli, które są popsute w
    konkretny sposób. Dzięki temu łatwiej zobaczyć, jak histogram PIT i worm
    plot reagują na bias średniej, złą wariancję i złą strukturę scoreline.


    Najważniejsza zasada: dla dobrze skalibrowanego modelu randomized PIT
    powinien wyglądać jak rozkład jednostajny na $[0, 1]$.
    """)
    return


@app.cell
def _():
    n_matches_slider = mo.ui.slider(
        start=500,
        stop=5000,
        step=500,
        value=1500,
        label="Liczba symulowanych meczów",
    )
    rho_slider = mo.ui.slider(
        start=-0.20,
        stop=0.10,
        step=0.01,
        value=-0.08,
        label="Prawdziwe rho Dixon-Coles",
    )
    seed_slider = mo.ui.number(value=20260430, label="Seed symulacji")
    mo.vstack([n_matches_slider, rho_slider, seed_slider])
    return n_matches_slider, rho_slider, seed_slider


@app.cell
def _():
    pit_variants = (
        "home_goals",
        "away_goals",
        "home_given_away",
        "away_given_home",
        "total_goals",
        "goal_difference",
    )
    pit_variant_groups = {
        "Marginal goals": ("home_goals", "away_goals"),
        "Conditional goals": ("home_given_away", "away_given_home"),
        "Total and difference": ("total_goals", "goal_difference"),
    }
    random_states = np.arange(10_000, 10_030, dtype=np.int64)
    max_goals_matrix = 6
    return max_goals_matrix, pit_variant_groups, pit_variants, random_states


@app.cell
def _(max_goals_matrix, n_matches_slider, rho_slider, seed_slider):
    rng = np.random.default_rng(int(seed_slider.value))
    n_matches = int(n_matches_slider.value)

    strength_home = rng.normal(loc=0.0, scale=0.33, size=n_matches)
    strength_away = rng.normal(loc=0.0, scale=0.33, size=n_matches)
    tempo = rng.normal(loc=0.0, scale=0.25, size=n_matches)
    lambda_home_true = np.exp(0.28 + tempo + strength_home)
    lambda_away_true = np.exp(-0.05 + tempo + strength_away)

    true_builder = PoissonMatrixBuilder(
        rho=float(rho_slider.value),
        max_goals_matrix=max_goals_matrix,
    )
    return lambda_away_true, lambda_home_true, rng, true_builder


@app.cell
def _(lambda_away_true, lambda_home_true, rng, true_builder):
    def sample_scorelines(lambda_home, lambda_away, matrix_builder, rng):
        home_scores = np.empty(lambda_home.shape[0], dtype=int)
        away_scores = np.empty(lambda_away.shape[0], dtype=int)
        for idx, (home_rate, away_rate) in enumerate(zip(lambda_home, lambda_away, strict=True)):
            matrix = matrix_builder.build_matrix(float(home_rate), float(away_rate))
            flat_index = rng.choice(matrix.size, p=matrix.ravel())
            home_scores[idx], away_scores[idx] = np.unravel_index(flat_index, matrix.shape)
        return home_scores, away_scores

    actual_home, actual_away = sample_scorelines(
        lambda_home_true,
        lambda_away_true,
        true_builder,
        rng,
    )
    simulated_matches = pd.DataFrame(
        {
            "lambda_home_true": lambda_home_true,
            "lambda_away_true": lambda_away_true,
            "home_score": actual_home,
            "away_score": actual_away,
            "total_goals": actual_home + actual_away,
            "goal_difference": actual_home - actual_away,
        }
    )
    return actual_away, actual_home, simulated_matches


@app.cell
def _(simulated_matches):
    mo.md(
        f"""
        **Wygenerowano {len(simulated_matches):,} meczów.**

        Średnie gole: home `{simulated_matches["home_score"].mean():.2f}`,
        away `{simulated_matches["away_score"].mean():.2f}`,
        total `{simulated_matches["total_goals"].mean():.2f}`.
        """.replace(",", " ")
    )
    return


@app.cell
def _(simulated_matches):
    mo.ui.dataframe(simulated_matches.head(20))
    return


@app.cell
def _(max_goals_matrix):
    @dataclass(frozen=True)
    class TemperedMatrixBuilder:
        """Sharpen or flatten a scoreline matrix while keeping the same mode area."""

        rho: float
        max_goals_matrix: int
        temperature: float

        def build_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
            base = PoissonMatrixBuilder(
                rho=self.rho,
                max_goals_matrix=self.max_goals_matrix,
            ).build_matrix(lambda_home, lambda_away)
            weighted = np.power(base, 1.0 / self.temperature)
            return weighted / weighted.sum()

    def poisson_builder(rho: float):
        return PoissonMatrixBuilder(rho=rho, max_goals_matrix=max_goals_matrix)

    def tempered_builder(rho: float, temperature: float):
        return TemperedMatrixBuilder(
            rho=rho,
            max_goals_matrix=max_goals_matrix,
            temperature=temperature,
        )

    return poisson_builder, tempered_builder


@app.cell
def _(
    lambda_away_true,
    lambda_home_true,
    poisson_builder,
    rho_slider,
    tempered_builder,
):
    true_rho = float(rho_slider.value)
    global_lambda_home = float(np.mean(lambda_home_true))
    global_lambda_away = float(np.mean(lambda_away_true))
    model_specs = {
        "Oracle": {
            "lambda_home": lambda_home_true,
            "lambda_away": lambda_away_true,
            "builder": poisson_builder(true_rho),
            "lesson": "Prawidłowa średnia i struktura zależności.",
        },
        "Global mean lambdas": {
            "lambda_home": np.full_like(lambda_home_true, global_lambda_home),
            "lambda_away": np.full_like(lambda_away_true, global_lambda_away),
            "builder": poisson_builder(true_rho),
            "lesson": "Jedna para lambd dla wszystkich meczów, bez heterogeniczności spotkań.",
        },
        "Home lambda too high": {
            "lambda_home": lambda_home_true * 1.25,
            "lambda_away": lambda_away_true,
            "builder": poisson_builder(true_rho),
            "lesson": "Model zawyża gole gospodarzy.",
        },
        "Away lambda too low": {
            "lambda_home": lambda_home_true,
            "lambda_away": lambda_away_true * 0.75,
            "builder": poisson_builder(true_rho),
            "lesson": "Model zaniża gole gości.",
        },
        "Overconfident": {
            "lambda_home": lambda_home_true,
            "lambda_away": lambda_away_true,
            "builder": tempered_builder(true_rho, temperature=0.65),
            "lesson": "Macierz jest zbyt skupiona wokół najbardziej prawdopodobnych wyników.",
        },
        "Underconfident": {
            "lambda_home": lambda_home_true,
            "lambda_away": lambda_away_true,
            "builder": tempered_builder(true_rho, temperature=1.55),
            "lesson": "Macierz jest zbyt płaska i daje za ciężkie ogony.",
        },
        "Wrong dependence": {
            "lambda_home": lambda_home_true,
            "lambda_away": lambda_away_true,
            "builder": poisson_builder(0.0),
            "lesson": "Marginesy są dobre, ale zależność niskich wyników jest błędna.",
        },
        "Wrong home advantage": {
            "lambda_home": lambda_home_true * 0.85,
            "lambda_away": lambda_away_true * 1.15,
            "builder": poisson_builder(true_rho),
            "lesson": "Tempo może wyglądać nieźle, ale przewaga stron jest przesunięta.",
        },
    }
    return (model_specs,)


@app.cell
def _(model_specs):
    interpretation_table = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "intentional_problem": spec["lesson"],
                "where_to_look_first": where,
            }
            for scenario, spec, where in [
                (
                    "Oracle",
                    model_specs["Oracle"],
                    "Wszystkie warianty powinny być blisko uniform.",
                ),
                (
                    "Global mean lambdas",
                    model_specs["Global mean lambdas"],
                    "home_goals i away_goals: globalnie może wyglądać poprawnie, lokalnie nie.",
                ),
                (
                    "Home lambda too high",
                    model_specs["Home lambda too high"],
                    "home_goals, goal_difference.",
                ),
                (
                    "Away lambda too low",
                    model_specs["Away lambda too low"],
                    "away_goals, goal_difference.",
                ),
                (
                    "Overconfident",
                    model_specs["Overconfident"],
                    "U-shape na histogramie, S-shape na worm plot.",
                ),
                (
                    "Underconfident",
                    model_specs["Underconfident"],
                    "Górka w środku histogramu PIT.",
                ),
                (
                    "Wrong dependence",
                    model_specs["Wrong dependence"],
                    "conditional goals i goal_difference.",
                ),
                (
                    "Wrong home advantage",
                    model_specs["Wrong home advantage"],
                    "goal_difference oraz oba marginesy.",
                ),
            ]
        ]
    )
    interpretation_table
    return


@app.cell
def _(actual_away, actual_home, model_specs, pit_variants, random_states):
    pit_results = {
        name: build_pit_diagnostics(
            lambda_home=spec["lambda_home"],
            lambda_away=spec["lambda_away"],
            actual_home=actual_home,
            actual_away=actual_away,
            matrix_builder=spec["builder"],
            variants=pit_variants,
            random_states=random_states,
            model_name=name,
            sample_name="simulated",
        )
        for name, spec in model_specs.items()
    }
    return (pit_results,)


@app.cell
def _(pit_results):
    summary_table = pd.concat(
        [result.summary.assign(scenario=name) for name, result in pit_results.items()],
        ignore_index=True,
    )[
        [
            "scenario",
            "pit_label",
            "n",
            "mean",
            "uniform_mean_delta",
            "std",
            "uniform_std_delta",
            "ks_statistic",
            "ks_pvalue",
        ]
    ]
    mo.ui.dataframe(summary_table)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Jak czytać histogramy

    Linia przerywana `1.0` to idealna gęstość uniform. Szary pas pokazuje
    przybliżony zakres losowych odchyleń dla próbki tej wielkości.

    - Więcej masy przy `0`: obserwacje są zwykle niżej niż model zakłada.
    - Więcej masy przy `1`: obserwacje są zwykle wyżej niż model zakłada.
    - Masy przy obu krańcach: model jest za pewny.
    - Górka w środku: model jest zbyt rozlany.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Histogramy goli gospodarzy i gości
    """)
    return


@app.cell
def _(pit_results, pit_variant_groups):
    histogram_marginal = plot_pit_histogram_replicates(
        pit_results,
        variants=pit_variant_groups["Marginal goals"],
        title="PIT histograms - marginal goal calibration",
        figsize=(12, 18),
    )
    histogram_marginal
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Histogramy warunkowych goli gospodarzy i gości
    """)
    return


@app.cell
def _(pit_results, pit_variant_groups):
    histogram_conditional = plot_pit_histogram_replicates(
        pit_results,
        variants=pit_variant_groups["Conditional goals"],
        title="PIT histograms - conditional scoreline structure",
        figsize=(12, 18),
    )
    histogram_conditional
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Histogramy łącznych goli i różnicy goli
    """)
    return


@app.cell
def _(pit_results, pit_variant_groups):
    histogram_total_diff = plot_pit_histogram_replicates(
        pit_results,
        variants=pit_variant_groups["Total and difference"],
        title="PIT histograms - total goals and goal difference",
        figsize=(12, 18),
    )
    histogram_total_diff
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Jak czytać worm ploty

    Worm plot pokazuje `empirical PIT quantile - theoretical uniform quantile`.
    Dla dobrego modelu mediana krzywych powinna trzymać się blisko zera i
    mieścić w szarym paśmie.

    Stałe przesunięcie krzywej sugeruje bias średniej. Kształt `S` albo
    odwrócone `S` częściej wskazuje na problem z rozproszeniem lub ogonami.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Worm ploty goli gospodarzy i gości
    """)
    return


@app.cell
def _(pit_results, pit_variant_groups):
    worm_marginal = plot_pit_worm_replicates(
        pit_results,
        variants=pit_variant_groups["Marginal goals"],
        title="Worm plots - marginal goal calibration",
        n_simulations=250,
        replicate_alpha=0.025,
        figsize=(12, 18),
    )
    worm_marginal
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Worm ploty warunkowych goli gospodarzy i gości
    """)
    return


@app.cell
def _(pit_results, pit_variant_groups):
    worm_conditional = plot_pit_worm_replicates(
        pit_results,
        variants=pit_variant_groups["Conditional goals"],
        title="Worm plots - conditional scoreline structure",
        n_simulations=250,
        replicate_alpha=0.025,
        figsize=(12, 18),
    )
    worm_conditional
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Worm ploty łącznych goli i różnicy goli
    """)
    return


@app.cell
def _(pit_results, pit_variant_groups):
    worm_total_diff = plot_pit_worm_replicates(
        pit_results,
        variants=pit_variant_groups["Total and difference"],
        title="Worm plots - total goals and goal difference",
        n_simulations=250,
        replicate_alpha=0.025,
        figsize=(12, 18),
    )
    worm_total_diff
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Interpretacje na pojedynczych przykładach

    Poniżej są te same diagnostyki, ale zawężone do pojedynczych scenariuszy i
    wariantów. To jest bardziej przydatne do nauki czytania kształtów niż duże
    siatki porównujące wszystko naraz.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Dobra kalibracja: `Oracle` dla `Home goals`

    To jest punkt odniesienia. Histogram powinien być blisko płaski wokół
    gęstości `1.0`, a worm plot powinien oscylować wokół zera i mieścić się
    głównie w szarym paśmie. Nie oczekujemy idealnej prostej, bo mamy skończoną
    próbkę i losowanie randomized PIT.
    """)
    return


@app.cell
def _(pit_results):
    oracle_home_histogram = plot_pit_histogram_replicates(
        {"Oracle": pit_results["Oracle"]},
        variants=("home_goals",),
        title="Good calibration example - Oracle - Home goals",
        figsize=(6.5, 4.6),
    )
    oracle_home_histogram
    return


@app.cell
def _(pit_results):
    oracle_home_worm = plot_pit_worm_replicates(
        {"Oracle": pit_results["Oracle"]},
        variants=("home_goals",),
        title="Good calibration example - Oracle - Home goals worm plot",
        n_simulations=500,
        replicate_alpha=0.04,
        figsize=(6.5, 4.6),
    )
    oracle_home_worm
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Lambda za duża: `Home lambda too high` dla `Home goals`

    Model przesuwa rozkład predykcyjny za wysoko, więc obserwowane gole
    gospodarzy częściej wypadają po lewej stronie rozkładu modelu. Histogram
    PIT powinien mieć większą masę przy małych wartościach i maleć w prawo.
    Na worm plocie mediana krzywych schodzi pod zero, często z kształtem
    zbliżonym do `U`.
    """)
    return


@app.cell
def _(pit_results):
    home_high_histogram = plot_pit_histogram_replicates(
        {"Home lambda too high": pit_results["Home lambda too high"]},
        variants=("home_goals",),
        title="Lambda too high example - Home goals histogram",
        figsize=(6.5, 4.6),
    )
    home_high_histogram
    return


@app.cell
def _(pit_results):
    home_high_worm = plot_pit_worm_replicates(
        {"Home lambda too high": pit_results["Home lambda too high"]},
        variants=("home_goals",),
        title="Lambda too high example - Home goals worm plot",
        n_simulations=500,
        replicate_alpha=0.04,
        figsize=(6.5, 4.6),
    )
    home_high_worm
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Lambda za mała: `Away lambda too low` dla `Away goals`

    Tu używamy wariantu gości, bo taki scenariusz jest zdefiniowany w symulacji.
    Model przesuwa rozkład predykcyjny za nisko, więc obserwowane gole gości
    częściej wypadają po prawej stronie rozkładu modelu. Histogram PIT powinien
    rosnąć w prawo. Na worm plocie mediana jest nad zerem, często z kształtem
    odwróconego `U`.
    """)
    return


@app.cell
def _(pit_results):
    away_low_histogram = plot_pit_histogram_replicates(
        {"Away lambda too low": pit_results["Away lambda too low"]},
        variants=("away_goals",),
        title="Lambda too low example - Away goals histogram",
        figsize=(6.5, 4.6),
    )
    away_low_histogram
    return


@app.cell
def _(pit_results):
    away_low_worm = plot_pit_worm_replicates(
        {"Away lambda too low": pit_results["Away lambda too low"]},
        variants=("away_goals",),
        title="Lambda too low example - Away goals worm plot",
        n_simulations=500,
        replicate_alpha=0.04,
        figsize=(6.5, 4.6),
    )
    away_low_worm
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Jedna para lambd dla wszystkich meczów: `Global mean lambdas`

    Tutaj model dostaje stałe wartości `lambda_home` i `lambda_away` równe
    średniej z całej próbki. Taki model ignoruje różnice pomiędzy meczami.
    Zbiorczo może wyglądać "znośnie", ale dla marginesów (`home_goals`,
    `away_goals`) zwykle ujawnia brak kalibracji zależnej od kontekstu meczu.
    """)
    return


@app.cell
def _(pit_results):
    global_lambdas_histogram = plot_pit_histogram_replicates(
        {"Global mean lambdas": pit_results["Global mean lambdas"]},
        variants=("home_goals", "away_goals"),
        title="Global mean lambdas - marginal goals histogram",
        figsize=(12, 4.8),
    )
    global_lambdas_histogram
    return


@app.cell
def _(pit_results):
    global_lambdas_worm = plot_pit_worm_replicates(
        {"Global mean lambdas": pit_results["Global mean lambdas"]},
        variants=("home_goals", "away_goals"),
        title="Global mean lambdas - marginal goals worm plot",
        n_simulations=500,
        replicate_alpha=0.04,
        figsize=(12, 4.8),
    )
    global_lambdas_worm
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Overconfident vs underconfident: `Total goals` i `Goal difference`

    Tu średnie lambdy są poprawne, ale sama macierz jest za ostra albo za płaska.
    Dla `Overconfident` histogram ma tendencję do kształtu `U`: skrajne PIT-y są
    za częste, bo prawdziwe wyniki częściej wpadają w ogony niż model zakłada.
    Dla `Underconfident` jest odwrotnie: skraje są za niskie, a środek za wysoki.

    `Goal difference` jest szczególnie ciekawy, bo worm ploty dla overconfidence
    i underconfidence potrafią mieć kształt `S`, ale w przeciwnych kierunkach.
    """)
    return


@app.cell
def _(pit_results):
    spread_histogram = plot_pit_histogram_replicates(
        {
            "Overconfident": pit_results["Overconfident"],
            "Underconfident": pit_results["Underconfident"],
        },
        variants=("total_goals", "goal_difference"),
        title="Spread examples - Total goals and goal difference histograms",
        figsize=(12, 7.5),
    )
    spread_histogram
    return


@app.cell
def _(pit_results):
    spread_worm = plot_pit_worm_replicates(
        {
            "Overconfident": pit_results["Overconfident"],
            "Underconfident": pit_results["Underconfident"],
        },
        variants=("total_goals", "goal_difference"),
        title="Spread examples - Total goals and goal difference worm plots",
        n_simulations=500,
        replicate_alpha=0.04,
        figsize=(12, 7.5),
    )
    spread_worm
    return


if __name__ == "__main__":
    app.run()
