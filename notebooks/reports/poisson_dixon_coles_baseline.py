import marimo

__generated_with = "0.19.10"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Model Bazowy (Baseline): Poisson Dixon-Coles dla 1. Ligi

    ## Cel Projektu
    Notatnik przedstawia budowę i ewaluację modelu statystycznego służącego do optymalizacji typów piłkarskich w konkursie "Supertyper". Celem jest maksymalizacja zdobywanych punktów według następującego klucza:
    * **3 pkt** - Dokładny wynik (np. typ 2:1, wynik 2:1)
    * **2 pkt** - Poprawna różnica bramek (np. typ 1:0, wynik 2:1)
    * **1 pkt** - Poprawne rozstrzygnięcie meczu (1X2)
    * **0 pkt** - Błędny typ

    ## Metodologia
    Jako model bazowy (Baseline) wykorzystano rozkład Poissona z korektą Dixona-Colesa (obsługującą zjawisko niedoszacowania remisów 0:0 i 1:1 w czystym rozkładzie Poissona).
    Ponieważ model na tym etapie nie uczy się na cechach historycznych, prawdopodobieństwa wejściowe (wartości oczekiwane goli) są estymowane na podstawie **kursów bukmacherskich** (rynek 1X2 oraz Under/Over 2.5). Wartości te są pozbawiane marży bukmacherskiej metodą potęgową (Power Implied Probabilities). Następnie z macierzy prawdopodobieństw wybierany jest wynik, który matematycznie maksymalizuje wartość oczekiwaną punktów (Expected Points) w konkursie.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Jak działa algorytm (Poisson Dixon-Coles)?

    Typowanie wyników opiera się na matematycznym zamodelowaniu prawdopodobieństw padnięcia konkretnych rezultatów bramkowych. Proces ten składa się z czterech głównych kroków:

    ### 1. Transformacja kursów na Lambdy
    Wartości oczekiwane liczby goli (tzw. lambdy) nie są estymowane z historycznych wyników, lecz odtwarzane (inżynieria odwrotna) z kursów bukmacherskich po usunięciu marży.

    Najpierw, na podstawie prawdopodobieństwa padnięcia powyżej 2.5 gola w meczu ($P_{>2.5}$), algorytm znajduje całkowitą oczekiwaną liczbę bramek ($\lambda_{total}$) przy użyciu metody numerycznej (optymalizacja Brenta), rozwiązując równanie:
    $$P_{>2.5} = 1 - \sum_{k=0}^{2} \frac{\lambda_{total}^k e^{-\lambda_{total}}}{k!}$$

    Następnie całkowita liczba bramek jest dzielona na drużynę gospodarzy i gości proporcjonalnie do ich szans na zwycięstwo (ignorując remis):
    $$\lambda_{home} = \lambda_{total} \cdot \frac{p_{home}}{p_{home} + p_{away}}$$
    $$\lambda_{away} = \lambda_{total} \cdot \frac{p_{away}}{p_{home} + p_{away}}$$

    ### 2. Bazowa Macierz Poissona
    Mając wyliczone lambdy, obliczana jest dwuwymiarowa macierz prawdopodobieństw dla każdego dokładnego wyniku (gdzie $x$ to gole gospodarzy, a $y$ to gole gości), przy początkowym założeniu, że liczba goli obu drużyn to zdarzenia niezależne:
    $$P(X=x, Y=y) = \frac{\lambda_{home}^x e^{-\lambda_{home}}}{x!} \cdot \frac{\lambda_{away}^y e^{-\lambda_{away}}}{y!}$$

    ### 3. Korekta Dixona-Colesa
    Czysty rozkład Poissona ma jedną wadę w modelowaniu piłki nożnej: zakłada pełną niezależność, przez co często niedoszacowuje wyników remisowych o niskiej liczbie bramek (0:0, 1:1) i przeszacowuje skromne zwycięstwa (1:0, 0:1).

    Aby to naprawić, wprowadzona zostaje funkcja korekcyjna $\tau(x,y)$ oparta na parametrze $\rho$, która modyfikuje bazowe prawdopodobieństwa dla wyników $x, y \in \{0, 1\}$:
    $$
    \tau(x,y) = \begin{cases}
    1 - \lambda_{home}\lambda_{away}\rho & \text{dla } x=0, y=0 \\
    1 + \lambda_{home}\rho & \text{dla } x=0, y=1 \\
    1 + \lambda_{away}\rho & \text{dla } x=1, y=0 \\
    1 - \rho & \text{dla } x=1, y=1 \\
    1 & \text{w pozostałych przypadkach}
    \end{cases}
    $$
    Ostateczne prawdopodobieństwo to $P_{DC}(x,y) = P(x,y) \cdot \tau(x,y)$. Ujemna wartość $\rho$ (np. -0.18 wykazana w Grid Search) matematycznie "zabiera" prawdopodobieństwo ze zwycięstw 1:0/0:1 i przenosi je na remisy 0:0 i 1:1.

    ### 4. Maksymalizacja Wartości Oczekiwanej (Expected Points)
    Model nie wybiera wyniku o najwyższym prawdopodobieństwie bezwzględnym. Zamiast tego dla każdej potencjalnej predykcji iteruje przez całą macierz możliwych wyników rzeczywistych i oblicza **Wartość Oczekiwaną Punktów** ($E[Points]$) według zasad Supertypera:
    $$E[Points_{pred}] = \sum_{x=0}^{max} \sum_{y=0}^{max} P_{DC}(x,y) \cdot S(pred_{home}, pred_{away}, x, y)$$
    Gdzie funkcja punktująca $S$ przyjmuje wartości:
    * **3 punkty** za dokładny wynik ($pred_{home} = x \land pred_{away} = y$),
    * **2 punkty** za poprawną różnicę bramek ($pred_{home} - pred_{away} = x - y$),
    * **1 punkt** za trafienie samego rozstrzygnięcia $1X2$ ($sgn(pred_{home} - pred_{away}) = sgn(x - y)$),
    * **0 punktów** w pozostałych przypadkach.

    Algorytm zwraca tę predykcję, dla której $E[Points]$ jest najwyższe.

    ---

    ## Metody agregacji kursów bukmacherskich
    Aby uodpornić model na wahania i błędy bukmacherów, w notatniku testowane są różne warianty estymacji prawdopodobieństw wejściowych:
    * **Średnie (Avg):** Zwykła średnia arytmetyczna z kursów dostępnych u wielu bukmacherów.
    * **Maksymalne (Max):** Najwyższy dostępny kurs na rynku na dany znak. Redukuje to wbudowaną w kursy marżę bukmacherską.
    * **Średnia obcięta (Trimmed Avg):** Średnia obliczana po odrzuceniu wartości skrajnych (maksymalnej i minimalnej) w celu eliminacji "szumu" oraz tzw. pewniaków-pułapek i błędów w wystawianiu linii przez analityków sportowych.
    """)
    return


@app.cell
def _():
    import sys
    import os

    # Dodaj katalog nadrzędny do ścieżki
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

    import marimo as mo

    return (mo,)


@app.cell
def _():
    from src.data import load_and_add_odds_columns_compact
    from src.features import add_power_implied_probabilities_standard_markets
    from src.models import (
        PoissonDixonColesModel,
        compute_points_per_match,
        evaluate_score_predictions,
        plot_predictions_summary,
        run_predictive_grid_search,
        plot_grid_search_2d,
        build_param_grid,
    )
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    return (
        PoissonDixonColesModel,
        add_power_implied_probabilities_standard_markets,
        build_param_grid,
        compute_points_per_match,
        load_and_add_odds_columns_compact,
        pd,
        plot_grid_search_2d,
        plot_predictions_summary,
        plt,
        run_predictive_grid_search,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Wczytanie danych
    """)
    return


@app.cell
def _(load_and_add_odds_columns_compact):
    df = load_and_add_odds_columns_compact()
    df
    return (df,)


@app.cell
def _(df, pd):
    start_day = pd.Timestamp("2025-08-04").tz_localize("Europe/Warsaw")

    df_current_season = df[df['season'] == 'current']
    df_current_season = \
        df_current_season[
            df_current_season['match_date'] >= start_day
        ]
    return (df_current_season,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Grid search
    """)
    return


@app.cell
def _(build_param_grid):
    param_grid = build_param_grid(
        {
            "rho": {"start": -0.30, "stop": 0.02, "step": 0.02},
            "bias_correction": {"start": 1.00, "stop": 1.32, "step": 0.02},
        }
    )
    return (param_grid,)


@app.cell
def _(PoissonDixonColesModel):
    def model_factory(**p):
        return PoissonDixonColesModel(**p)

    return (model_factory,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Średnie kursy
    """)
    return


@app.cell
def _(add_power_implied_probabilities_standard_markets, df):
    df_avg_odds = add_power_implied_probabilities_standard_markets(df, odds_prefix='avg')
    df.isnull().sum()
    return (df_avg_odds,)


@app.cell
def _(df_avg_odds, model_factory, param_grid, run_predictive_grid_search):
    search_avg_odds = run_predictive_grid_search(
        model_factory=model_factory,
        param_grid=param_grid,
        df=df_avg_odds,
        cache_mode="use"
    ) 
    return (search_avg_odds,)


@app.cell
def _(plot_grid_search_2d, plt, search_avg_odds):
    ax_avg_odds = plot_grid_search_2d(
        search_avg_odds.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="avg_points"
    )
    # Zmiana rozmiaru: (szerokość, wysokość) w calach
    _fig = plt.gcf()
    _fig.set_size_inches(12, 6) 

    # Odświeżenie widoku 
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Odcięte średnie kursy
    """)
    return


@app.cell
def _(add_power_implied_probabilities_standard_markets, df):
    df_avg_trimmed_odds = add_power_implied_probabilities_standard_markets(df, odds_prefix='trimmed_avg')
    df_avg_trimmed_odds.dropna(inplace = True)
    df_avg_trimmed_odds.isnull().sum().sum()
    return (df_avg_trimmed_odds,)


@app.cell
def _(
    df_avg_trimmed_odds,
    model_factory,
    param_grid,
    run_predictive_grid_search,
):
    search_avg_trimmed_odds = run_predictive_grid_search(
        model_factory=model_factory,
        param_grid=param_grid,
        df=df_avg_trimmed_odds,
        cache_mode="use"
    ) 
    return (search_avg_trimmed_odds,)


@app.cell
def _(plot_grid_search_2d, plt, search_avg_trimmed_odds):
    ax_avg_trimmed_odds = plot_grid_search_2d(
        search_avg_trimmed_odds.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="avg_points"
    )

    # Odświeżenie widoku 
    # Zmiana rozmiaru: (szerokość, wysokość) w calach
    _fig = plt.gcf()
    _fig.set_size_inches(12, 6) 

    # Odświeżenie widoku 
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Maksymalne kursy
    """)
    return


@app.cell
def _(add_power_implied_probabilities_standard_markets, df):
    df_max_odds = add_power_implied_probabilities_standard_markets(df, odds_prefix='max')
    df_max_odds.dropna(inplace=True)
    df_max_odds.isnull().sum().sum()
    return (df_max_odds,)


@app.cell
def _(df_max_odds, model_factory, param_grid, run_predictive_grid_search):
    search_max_odds = run_predictive_grid_search(
        model_factory=model_factory,
        param_grid=param_grid,
        df=df_max_odds,
        cache_mode="use"
    ) 
    return (search_max_odds,)


@app.cell
def _(plot_grid_search_2d, plt, search_max_odds):
    ax_max_odds = plot_grid_search_2d(
        search_max_odds.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="avg_points"
    )

    # Odświeżenie widoku 
    # Zmiana rozmiaru: (szerokość, wysokość) w calach
    _fig = plt.gcf()
    _fig.set_size_inches(12, 6) 

    # Odświeżenie widoku 
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Grid search skupione na miejscu ekstremum z mniejszym krokiem na najlepszych kursach
    """)
    return


@app.cell
def _(build_param_grid):
    param_grid_in_extreme_avg = build_param_grid(
        {
            "rho": {"start": -0.20, "stop": -0.16, "step": 0.005},
            "bias_correction": {"start": 1.00, "stop": 1.04, "step": 0.005},
        }
    )
    return (param_grid_in_extreme_avg,)


@app.cell
def _(
    df_avg_odds,
    model_factory,
    param_grid_in_extreme_avg,
    run_predictive_grid_search,
):
    search_avg_odds_in_extreme = run_predictive_grid_search(
        model_factory=model_factory,
        param_grid=param_grid_in_extreme_avg,
        df=df_avg_odds,
        cache_mode="use"
    ) 
    return (search_avg_odds_in_extreme,)


@app.cell
def _(plot_grid_search_2d, plt, search_avg_odds_in_extreme):
    ax_avg_odds_in_extreme = plot_grid_search_2d(
        search_avg_odds_in_extreme.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="avg_points"
    )
    # Odświeżenie widoku 
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Wnioski z optymalizacji (Grid Search) i Stabilność Ekstremum

    Testy różnych metod agregacji kursów (Avg, Max, Trimmed Avg) pokazały zbliżone wartości maksymalne punktów, jednak dla różnych zestawów parametrów. Wybór kursów **uśrednionych (Avg)** podyktowany był analizą stabilności na mapie ciepła – ekstremum dla tych kursów znajdowało się w szerszym "płaskowyżu", gdzie sąsiednie komórki (różniące się nieznacznie parametrami) również dawały wysokie i stabilne wyniki.

    Po zawężeniu siatki poszukiwań (mniejszy krok) w okolicy tego ekstremum, optymalne parametry dla modelu historycznego ukształtowały się na poziomie:
    * **`rho`** $\approx -0.18$
    * **`bias_correction`** $\approx 1.035$

    Taka kombinacja lekko podbija ogólną liczbę bramek w meczu i jednocześnie silnie koryguje szanse na niskie remisy.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Test w aktualnym sezonie na najlepszych parametrach i kursach
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Większy krok i większy zakres
    """)
    return


@app.cell
def _(add_power_implied_probabilities_standard_markets, df_current_season):
    # wiem że tutaj nie ma braków danych
    df_avg_odds_current = add_power_implied_probabilities_standard_markets(df_current_season, odds_prefix='avg')
    return (df_avg_odds_current,)


@app.cell
def _(
    df_avg_odds_current,
    model_factory,
    param_grid,
    run_predictive_grid_search,
):
    search_avg_odds_current = run_predictive_grid_search(
        model_factory=model_factory,
        param_grid=param_grid,
        df=df_avg_odds_current,
        score_key="total_points",
        cache_mode="use"
    ) 
    return (search_avg_odds_current,)


@app.cell
def _(plot_grid_search_2d, plt, search_avg_odds_current):
    ax_avg_odds_current = plot_grid_search_2d(
        search_avg_odds_current.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="total_points"
    )

    # Odświeżenie widoku 
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mniejszy krok i mniejszy zakres
    """)
    return


@app.cell
def _(
    df_avg_odds_current,
    model_factory,
    param_grid_in_extreme_avg,
    run_predictive_grid_search,
):
    search_avg_odds_current_in_potential_extreme = run_predictive_grid_search(
        model_factory=model_factory,
        param_grid=param_grid_in_extreme_avg,
        df=df_avg_odds_current,
        score_key="total_points",
        cache_mode="use"
    ) 
    return (search_avg_odds_current_in_potential_extreme,)


@app.cell
def _(plot_grid_search_2d, plt, search_avg_odds_current_in_potential_extreme):
    ax_avg_odds_current_in_potential_extreme = plot_grid_search_2d(
        search_avg_odds_current_in_potential_extreme.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="total_points"
    )

    # Odświeżenie widoku 
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Zderzenie z rzeczywistością: Overfitting vs Generalizacja w konkursie

    Analizując wyniki dla aktualnego sezonu (połowa rozgrywek):
    * **Mój wynik manualny:** 151 punktów
    * **Wynik Lidera konkursu:** 154 punkty
    * **Wynik Modelu (historycznie najlepsze parametry):** 146 punktów

    **Dyskusja o przeuczeniu (Overfitting):**
    Jak widać na siatce Grid Search wyłącznie dla *obecnego sezonu*, model teoretycznie mógłby zdobyć równe 154 punkty (doganiając lidera), gdybyśmy dobrali parametry idealnie pod te konkretne 163 mecze. Byłoby to jednak klasyczne **przeuczenie modelu (overfitting)** do szumu i specyfiki zaledwie jednej rundy.

    Parametry wybrane na podstawie wielosezonowej historii (`rho` = -0.18, `bias_correction` = 1.035) dają "tylko" 146 punktów, ale są **odporne na wariancję** i dają największą szansę na stabilne punktowanie w długim terminie (w rundzie wiosennej). Różnica 5 punktów między moją intuicją (151 pkt) a modelem (146 pkt) wynika głównie z omówionego wcześniej błędu przeszacowania faworytów przez bukmacherów, który koryguję manualnie.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tabela z wynikami
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Potencjalnie najlepszy model
    """)
    return


@app.cell
def _(PoissonDixonColesModel):
    potential_best_model = PoissonDixonColesModel(bias_correction=1.035, rho=-0.18)
    return (potential_best_model,)


@app.cell
def _(df_avg_odds_current, potential_best_model):
    df_potential_best = potential_best_model.predict(df_avg_odds_current)
    return (df_potential_best,)


@app.cell
def _(df_potential_best):
    df_potential_best
    return


@app.cell
def _(compute_points_per_match, df_potential_best):
    df_potential_best['points_score'] = compute_points_per_match(df_potential_best)
    return


@app.cell
def _(df_potential_best):
    df_potential_best.info()
    return


@app.cell
def _(df_potential_best):
    _columns = [
        'match_date',
        'home_team',
        'away_team',
        'avg_1',
        'avg_X',
        'avg_2',
        'home_score',
        'away_score',
        'pred_score',
        'points_score',
        'pred_xpts',
        'exp_goals_home',
        'exp_goals_away',
    ]
    df_potential_best[_columns]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Analiza wyników predykcji
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Segmentacja błędów i Macierz Pomyłek 1X2

    Raport z Anomalii: Przewidywane '1', Rzeczywiste 'X'
    Powyższa macierz pomyłek ujawnia systematyczny błąd modelu – aż w 30 przypadkach model typował zwycięstwo gospodarzy, podczas gdy w rzeczywistości padał remis.
    """)
    return


@app.cell
def _(df_potential_best, plot_predictions_summary):
    plot_predictions_summary(df_potential_best, model_name="Poisson Dixon-Coles (baseline)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Tabela "Złodziei Punktów" (Model vs Zespoły)
    Poniższe zestawienie agreguje skuteczność modelu w zależności od grającej drużyny.
    * **Dół tabeli:** Drużyny, na których model zarabia najmniej punktów (duża wariancja, gra wbrew kursom, niespodzianki).
    * **Góra tabeli:** Drużyny bardzo przewidywalne dla bukmacherów i modelu statystycznego.
    """)
    return


@app.cell
def _(df_potential_best, pd):
    # Lista tymczasowa do zbudowania DataFrame'u
    _team_performance = []

    for _idx, _row in df_potential_best.iterrows():
        _team_performance.append({
            'Team': _row['home_team'],
            'Points': _row['points_score'],
            'Role': 'Home'
        })
        _team_performance.append({
            'Team': _row['away_team'],
            'Points': _row['points_score'],
            'Role': 'Away'
        })

    df_teams = pd.DataFrame(_team_performance)

    team_stats = df_teams.groupby('Team')['Points'].agg(
        Matches='count',
        Total_Points='sum',
        Avg_Points='mean'
    ).sort_values(by='Avg_Points', ascending=True)

    team_stats.reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Podsumowanie i Next Steps (Kierunek v2)

    Obecny model radzi sobie solidnie jak na proste podejście generatywne – punktuje w dużej części spotkań i daje stabilną bazę 146 punktów w jednej rundzie. Osiągnęliśmy jednak "sufit" możliwości wynikający z faktu, że model opiera się **wyłącznie na kursach bukmacherskich**.

    Główne ograniczenia obecnego podejścia:
    1. **Niewolnictwo marży:** Model jest podatny na manipulacje kursami przez analityków sportowych (np. zaniżanie kursów na medialnych faworytów).
    2. **Brak pamięci o formie:** Poisson-Dixon-Coles w tej formie nie wie, że drużyna "X" grała trzy dni temu wyczerpujący mecz w Pucharze Polski, ani że ich najlepszy napastnik leczy kontuzję.

    ## Następny krok: Przejście z modelu generatywnego na predykcyjny
    Aby uniezależnić się od błędów bukmacherów, w kolejnej iteracji zastosowane zostanie podejście opierające się na **uczeniu maszynowym (Machine Learning)**:
    * Algorytmem docelowym będzie **XGBoost**.
    * Model nie będzie korzystał ze z góry wyliczonych lambd, lecz samodzielnie je przewidywał na podstawie nowo utworzonych cech historycznych (**Feature Engineering**), takich jak: forma krocząca z 5 meczów, kalendarz, pozycja w tabeli.
    * Wdrożona zostanie rygorystyczna **walidacja krocząca w czasie (Walk-Forward Validation)**, aby precyzyjniej symulować warunki konkursowe.
    """)
    return


if __name__ == "__main__":
    app.run()
