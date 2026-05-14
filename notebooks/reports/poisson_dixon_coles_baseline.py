import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    import sys
    import os

    import marimo as mo

    from src.data import load_and_add_odds_columns_compact
    from src.features import add_power_implied_probabilities_standard_markets
    from src.models import (
        PoissonDixonColesModel,
        compute_points_per_match,
        evaluate_score_predictions,
        plot_predictions_summary,
        run_predictive_grid_search,
        run_predictive_nll_grid_search,
        run_predictive_points_weighted_nll_grid_search,
        plot_grid_search_2d,
        build_param_grid,
        plot_nll_grid_search_2d,
        plot_predictions_scoreline_summary,
    )

    from src.models.evaluation.pit import (
        build_pit_diagnostics,
        plot_pit_histogram_replicates,
        plot_pit_worm_replicates,
    )

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    import numpy as np


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Model Bazowy (Baseline): Poisson Dixon-Coles dla 1. Ligi

    _Ten notatnik znajduje się w repozytorium w ścieżce:_
    `notebooks/reports/poisson_dixon_coles_baseline.py`

    ## Cel Projektu
    Notatnik przedstawia budowę i ewaluację modelu statystycznego służącego do optymalizacji typów piłkarskich w konkursie "Supertyper". Celem jest maksymalizacja zdobywanych punktów według następującego klucza:
    * **3 pkt** - Dokładny wynik (np. typ 2:1, wynik 2:1)
    * **2 pkt** - Poprawna różnica bramek (np. typ 1:0, wynik 2:1)
    * **1 pkt** - Poprawne rozstrzygnięcie meczu (1X2)
    * **0 pkt** - Błędny typ

    ## Streszczenie

    - **Punkt wyjścia:** Zbudowano bazowy model predykcyjny (Poisson z korektą Dixona-Colesa) dla wyników 1. Ligi, oparty wyłącznie na prawdopodobieństwach odtworzonych z kursów bukmacherskich.

    - **Optymalizacja pod punkty (avg_points):** Przeszukiwanie siatki pod kątem maksymalizacji punktów konkursowych daje zadowalające rezultaty w praktyce, jednak powierzchnia tej metryki jest bardzo niestabilna, co utrudnia wskazanie trwałego optimum.

    - **Problem z metrykami probabilistycznymi:** Standardowe NLL (Negative Log-Likelihood) wykazuje dużą stabilność, ale nie daje najlepszych wyników konkursowych, co wynika ze zbyt małej elastyczności obecnego modelu (tylko jeden globalny parametr korygujący dla obu drużyn). Z kolei Ważone NLL (Weighted NLL) całkowicie zawodzi – optymalizator "obchodzi" metrykę, drastycznie kompresując masę prawdopodobieństwa na najniższych wynikach (np. 1:0), co niszczy kalibrację i odrywa model od rzeczywistości.

    - **Metoda agregacji kursów:** Stwierdzono, że sposób agregacji kursów (średnia vs. średnia obcięta vs. kursy maksymalne) ma marginalny wpływ na kształt optimum parametrów. Zalecono stosowanie średniej obciętej jako zabezpieczenia przed skrajnymi błędami bukmacherów.

    - **Dalsze kroki:** Głównym wnioskiem jest konieczność odejścia od jednego globalnego mnożnika na rzecz osobnego modelowania siły ataku i obrony (np. za pomocą modeli GLM) oraz weryfikacja zachowania funkcji kosztu NLL na danych syntetycznych.

    ---

    ## Metodologia
    Jako model bazowy (Baseline) wykorzystano rozkład Poissona z korektą Dixona-Colesa (obsługującą zjawisko niedoszacowania remisów 0:0 i 1:1 w czystym rozkładzie Poissona).
    Ponieważ model na tym etapie nie uczy się na cechach historycznych, prawdopodobieństwa wejściowe (wartości oczekiwane goli) są estymowane na podstawie **kursów bukmacherskich** (rynek 1X2 oraz Under/Over 2.5). Wartości te są pozbawiane marży bukmacherskiej metodą potęgową (Power Implied Probabilities). Następnie z macierzy prawdopodobieństw wybierany jest wynik, który matematycznie maksymalizuje wartość oczekiwaną punktów (Expected Points) w konkursie.
    """)
    return


@app.cell(hide_code=True)
def _():
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

    ## Cel grid search
    W dalszej części raportu dla każdej metody agregacji kursów wykonywany jest grid search po dwóch parametrach modelu: `bias_correction` oraz `rho`. Wyniki są pokazywane na heatmapach, żeby nie tylko wskazać najlepszy punkt siatki, ale też zobaczyć, jak stabilnie zachowują się metryki w sąsiedztwie optimum i czy różne metryki prowadzą do podobnych decyzji.

    Taki układ celowo opiera się na prostym modelu. Dzięki temu łatwiej oddzielić wpływ samych kursów bukmacherskich od wpływu dodatkowych cech lub bardziej złożonego uczenia maszynowego. Parametr `bias_correction` działa jak globalna korekta oczekiwanej liczby bramek. Bazowe lambdy odtworzone z kursów są przed zbudowaniem rozkładu Poissona mnożone przez ten współczynnik:

    $$\lambda_{home}^{model} = \text{bias\_correction} \cdot \lambda_{home}^{base}$$
    $$\lambda_{away}^{model} = \text{bias\_correction} \cdot \lambda_{away}^{base}$$

    Wartości powyżej `1.0` zwiększają lambdy obu drużyn i przesuwają rozkład w stronę wyższych wyników, a wartości poniżej `1.0` obniżają oczekiwaną liczbę bramek i wzmacniają scenariusze z mniejszą liczbą goli. Parametr `rho` kontroluje korektę Dixona-Colesa dla niskich wyników, szczególnie `0:0`, `1:0`, `0:1` i `1:1`, czyli obszaru, w którym niezależne rozkłady Poissona często są gorzej skalibrowane.

    Porównanie heatmap dla `avg_points`, `NLL`, `weighted NLL` oraz diagnostyk PIT pozwala sprawdzić, czy parametry dobre z perspektywy punktów konkursowych są również dobrze skalibrowane probabilistycznie.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Dane

    Raport dotyczy meczów 1. Ligi. Dane obejmują wyniki spotkań oraz kursy bukmacherskie dla rynków wykorzystywanych do odtworzenia prawdopodobieństw wejściowych modelu.

    Na potrzeby analizy dane są dzielone na dwie części. Sezony `2020/2021`-`2024/2025` tworzą próbę historyczną używaną do grid search i wyboru parametrów. Etykieta `current` oznacza aktualny sezon projektu, czyli `2025/2026`. Z tego sezonu bierzemy tylko mecze od `2025-08-04`, ponieważ od tej daty rozpoczął się konkurs Supertyper i dopiero te spotkania można sensownie porównywać z punktacją innych uczestników.

    Aktualny sezon nie jest pełną walidacją produkcyjną modelu. Pełni raczej rolę praktycznego testu porównawczego: pozwala sprawdzić, jak model wypada względem realnych typów graczy w konkursie i czy jego zachowanie jest na tyle słabe lub niestabilne, że dany wariant warto odrzucić.
    """)
    return


@app.cell
def _():
    df = load_and_add_odds_columns_compact()
    df
    return (df,)


@app.cell
def _(df):
    _nulls = df.isnull().sum()  
    print(_nulls[_nulls > 0]) # braki danych
    return


@app.cell
def _(df):
    df_grid_search = df
    df_grid_search = df_grid_search[~((df_grid_search["season"].eq("current")))]
    print(df_grid_search['season'].unique())
    return (df_grid_search,)


@app.cell
def _(df):
    start_day = pd.Timestamp("2025-08-04").tz_localize("Europe/Warsaw")

    df_current_season = df[df['season'] == 'current']
    df_current_season = \
        df_current_season[
            df_current_season['match_date'] >= start_day
        ]
    return (df_current_season,)


@app.cell
def _(df_current_season, df_grid_search):
    historical_seasons = sorted(df_grid_search["season"].unique())
    current_start_day = pd.Timestamp("2025-08-04").tz_localize("Europe/Warsaw")

    data_summary_md = mo.md(f"""
    ### Podział próby

    - **Próba historyczna:** sezony `{', '.join(historical_seasons)}`; liczba meczów: **{len(df_grid_search)}**.
    - **Aktualny sezon konkursowy:** `2025/2026` (`season == "current"`), od `{current_start_day.date()}`; liczba meczów: **{len(df_current_season)}**.
    - **Cel podziału:** parametry dobieramy na historii, a aktualny sezon służy do porównania modelu z uczestnikami konkursu.
    """)
    data_summary_md
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Grid search

    Grid search służy tutaj do systematycznego sprawdzenia, jak model zachowuje się dla różnych kombinacji dwóch parametrów: `bias_correction` oraz `rho`. Dla każdego punktu siatki model generuje predykcje na tej samej próbie historycznej, a następnie wynik jest oceniany kilkoma metrykami. Heatmapy pozwalają zobaczyć nie tylko najlepszą kombinację parametrów, ale też kształt całej powierzchni metryki: czy optimum jest stabilne, czy leży na brzegu siatki, oraz czy różne kryteria wskazują podobny obszar.

    Pierwszą metryką jest `avg_points`, czyli średnia liczba punktów zdobywanych według zasad Supertypera. To metryka najbardziej zgodna z celem konkursowym, więc dla niej wyższa wartość jest lepsza.

    Drugą metryką jest **negative log-likelihood** rzeczywistego wyniku. Dla meczu `i` niech `(h_i, a_i)` oznacza rzeczywisty wynik, a `P_θ(x, y)` prawdopodobieństwo wyniku `(x, y)` z macierzy modelu dla parametrów `θ = (rho, bias_correction)`. Wtedy:

    $$
    NLL(θ) = -\frac{1}{n}\sum_{i=1}^{n}\log P_θ(h_i, a_i)
    $$

    Niższy NLL oznacza, że model przypisywał większe prawdopodobieństwo wynikom, które faktycznie wystąpiły. To jest metryka kalibracji probabilistycznej, a nie bezpośrednio punktacji konkursowej.

    Trzeci wariant to **weighted NLL**, który zamiast patrzeć wyłącznie na dokładny wynik, liczy ważoną masę prawdopodobieństwa wokół rzeczywistego wyniku zgodnie z logiką punktacji `3/2/1`. Dokładny wynik ma wagę `1`, wyniki z tą samą różnicą bramek wagę `2/3`, a wyniki z tym samym rozstrzygnięciem 1X2 wagę `1/3`.

    Dla czytelności zdefiniujmy dwa zbiory wyników względem rzeczywistego wyniku `(h_i, a_i)`:

    $$
    D_i = \{(x, y): x - y = h_i - a_i,\ (x, y) \ne (h_i, a_i)\}
    $$

    $$
    O_i = \{(x, y): \operatorname{sgn}(x-y) = \operatorname{sgn}(h_i-a_i),\ x-y \ne h_i-a_i\}
    $$

    Wtedy ważona masa prawdopodobieństwa ma prostszy zapis:

    $$ p_i^{weighted}(θ) = P_θ(h_i, a_i) + \frac{2}{3}\sum_{(x,y)\in D_i} P_θ(x, y) + \frac{1}{3}\sum_{(x,y)\in O_i} P_θ(x, y) $$

    $$
    weighted\ NLL(θ) = -\frac{1}{n}\sum_{i=1}^{n}\log p_i^{weighted}(θ)
    $$

    Weighted NLL jest więc próbą połączenia perspektywy probabilistycznej z tym, jak konkurs nagradza predykcje bliskie rzeczywistemu wynikowi. Podobnie jak dla zwykłego NLL, niższa wartość jest lepsza.
    """)
    return


@app.cell
def _():
    param_grid = build_param_grid(
        {
            "rho": {"start": -0.30, "stop": 0.06, "step": 0.02},
            "bias_correction": {"start": 0.94, "stop": 1.24, "step": 0.02},
        }
    )
    return (param_grid,)


@app.function
def model_factory(**p):
    return PoissonDixonColesModel(**p, use_over25_interpolation=True)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Średnie kursy

    W tej części raportu wejściem do modelu są kursy zagregowane jako **średnia arytmetyczna** (`avg_*`) po bukmacherach, z odtworzeniem prawdopodobieństw i lambd jak w metodologii na początku notatnika. To **główna ścieżka** porównań: na tych danych wykonujemy pełny cykl grid search pod **`avg_points`**, **NLL** oraz **weighted NLL** (w tym dodatkową siatkę), bo na nich opieramy też dalszą diagnostykę i zrzuty predykcji.
    """)
    return


@app.cell
def _(df_grid_search):
    df_avg_odds = add_power_implied_probabilities_standard_markets(df_grid_search, odds_prefix='avg')
    return (df_avg_odds,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Grid search po średnich kursach

    Uruchamiana jest w **`param_grid`**: dla każdej pary `(bias_correction, ρ)` liczone są predykcje na próbie historycznej i ocena **kolejnymi metrykami** (najpierw niestabilne **`avg_points`**, potem **`NLL`**, **`weighted NLL`** i na końcu **`avg_points`** na siatce przesuniętej pod weighted NLL). Kolejne podsekcje to ten sam eksperyment na **`df_avg_odds`** — zmienia się tylko funkcja celu i zakres siatki tam, gdzie to zaznaczono.
    """)
    return


@app.cell
def _(df_avg_odds, param_grid):
    search_avg_odds = run_predictive_grid_search(
        model_factory=model_factory,
        param_grid=param_grid,
        df=df_avg_odds,
        cache_mode="use"
    ) 
    return (search_avg_odds,)


@app.cell
def _(search_avg_odds):
    ax_avg_odds = plot_grid_search_2d(
        search_avg_odds.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="avg_points"
    )
    # Zmiana rozmiaru: (szerokość, wysokość) w calach
    _fig = plt.gcf()
    _fig.set_size_inches(12, 6) 

    mo.output.append(ax_avg_odds)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja:** Ta metryka jest wyraźnie niestabilna. Niewielka zmiana parametrów potrafi mocno zmienić wartość średniego punktowania, dlatego trudno na jej podstawie wskazać jedno wiarygodne optimum. Tę heatmapę warto traktować przede wszystkim jako punkt odniesienia dla pozostałych metryk: sprawdzamy, czy obszary preferowane przez NLL lub weighted NLL pokrywają się z miejscami, gdzie model dobrze punktuje w konkursie.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Ten sam układ (ρ, bias), ranking po NLL wyniku

    Ta heatmapa pokazuje tę samą siatkę parametrów i te same średnie kursy, ale ranking jest oparty na `NLL`. Niższa wartość oznacza lepsze przypisanie prawdopodobieństwa rzeczywistym wynikom, optimum może znajdować się gdzie indziej niż dla `avg_points`.
    """)
    return


@app.cell
def _(df_avg_odds, param_grid):
    search_avg_odds_nll = run_predictive_nll_grid_search(
        model_factory=model_factory,
        param_grid=param_grid,
        df=df_avg_odds,
        cache_mode="use",
    )
    return (search_avg_odds_nll,)


@app.cell
def _(search_avg_odds_nll):
    ax_avg_odds_nll = plot_nll_grid_search_2d(
        search_avg_odds_nll.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="objective_metric",
    )
    _fig = plt.gcf()
    _fig.set_size_inches(12, 6)
    mo.output.append(ax_avg_odds_nll)
    return


@app.cell
def _(search_avg_odds_nll):
    mo.md(f"""
    Najlepsze parametry wybrane na podstawie Grid Search NLL dla średnich kursów:
    - rho: {np.round(search_avg_odds_nll.best_params['rho'], 2)}
    - bias_correction: {np.round(search_avg_odds_nll.best_params['bias_correction'], 2)}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja:** Dla NLL widać dużo stabilniejsze ekstremum na relatywnie gładkiej powierzchni. Najlepszy obszar znajduje się w okolicy `rho = 0` oraz `bias_correction = 1.0`, czyli blisko modelu bez dodatkowego przesuwania bazowych lambd i bez silnej korekty Dixona-Colesa.

    Warto zauważyć, że optymalne `rho` z tego grid search nie jest zgodne z klasyczną intuicją dla piłki nożnej, gdzie zwykle oczekiwalibyśmy wartości ujemnej, mniej więcej w przedziale `-0.10` do `-0.20`. Może to oznaczać, że jeden globalny parametr `bias_correction` nie wystarcza do pełnej poprawy kalibracji lambd odtworzonych z kursów. Jednocześnie najlepszy wariant według NLL był tylko przeciętny pod względem punktowania: na wcześniejszej siatce osiągał średnio około `0.792` punktu, czyli jedną ze słabszych wartości dla metryki `avg_points`.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Ten sam układ (ρ, bias), ranking po ważonym NLL wyniku

    Ta heatmapa pokazuje wynik `weighted NLL` na pierwotnej siatce parametrów. Metryka nadal jest minimalizowana, ale bierze pod uwagę nie tylko dokładny wynik, lecz także masę prawdopodobieństwa na wynikach punktujących częściowo w konkursie.
    """)
    return


@app.cell
def _(df_avg_odds, param_grid):
    search_avg_odds_weighted_nll = run_predictive_points_weighted_nll_grid_search(
        model_factory=model_factory,
        param_grid=param_grid,
        df=df_avg_odds,
        cache_mode="use",
    )
    return (search_avg_odds_weighted_nll,)


@app.cell
def _(search_avg_odds_weighted_nll):
    ax_avg_odds_weighted_nll = plot_nll_grid_search_2d(
        search_avg_odds_weighted_nll.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="objective_metric",
    )
    _fig = plt.gcf()
    _fig.set_size_inches(12, 6)
    mo.output.append(ax_avg_odds_weighted_nll)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja:** Dla weighted NLL najlepsza wartość wypada na krańcu pierwotnej siatki, w lewym dolnym rogu. To sugeruje, że badany zakres parametrów nie obejmuje jeszcze właściwego ekstremum tej metryki. Z tego powodu potrzebna jest dodatkowa siatka przesunięta w stronę niższego `bias_correction` i wyższego `rho`, żeby sprawdzić, gdzie weighted NLL faktycznie osiąga optimum.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Dodatkowa siatka wokół optimum weighted NLL

    Ponieważ weighted NLL na pierwotnej siatce osiągało najlepszą wartość na jej krańcu, wykonywany jest dodatkowy grid search w przesuniętym zakresie parametrów. Celem nie jest automatyczne przyjęcie bardziej ekstremalnych parametrów, tylko sprawdzenie, czy metryka faktycznie dalej poprawia się poza początkowym zakresem oraz jak wygląda kształt powierzchni w tym obszarze.

    Na tej samej dodatkowej siatce pokazane jest również `avg_points`. To zestawienie odpowiada na praktyczne pytanie: czy obszar preferowany przez weighted NLL jest jednocześnie korzystny dla średniego punktowania w konkursie, czy poprawa metryki probabilistycznej odbywa się kosztem celu konkursowego.
    """)
    return


@app.cell
def _(df_avg_odds):
    param_grid_weighted_nll = build_param_grid(
        {
            "rho": {"start": 0.42, "stop": 0.80, "step": 0.02},
            "bias_correction": {"start": 0.34, "stop": 0.56, "step": 0.02},
        }
    )

    search_avg_odds_weighted_nll_extreme_grid = run_predictive_points_weighted_nll_grid_search(
        model_factory=model_factory,
        param_grid=param_grid_weighted_nll,
        df=df_avg_odds,
        cache_mode="use",
    )
    return param_grid_weighted_nll, search_avg_odds_weighted_nll_extreme_grid


@app.cell
def _(search_avg_odds_weighted_nll_extreme_grid):
    _ax = plot_nll_grid_search_2d(
        search_avg_odds_weighted_nll_extreme_grid.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="objective_metric",
    )
    _fig = plt.gcf()
    _fig.set_size_inches(12, 6)
    mo.output.append(_ax)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja:** Na dodatkowej siatce okazuje się, że dla weighted NLL najlepsze parametry to około `rho = 0.68` oraz `bias_correction = 0.46`. Są to wartości bardzo nietypowe z punktu widzenia interpretacji modelu: `rho` jest silnie dodatnie, choć zgodnie z klasyczną intuicją dla korekty Dixona-Colesa powinno raczej być ujemne, a bazowe lambdy są zmniejszane o ponad połowę. Taki wynik trzeba więc traktować ostrożnie: metryka znajduje optimum matematyczne, ale parametry są trudne do uzasadnienia modelowo.
    """)
    return


@app.cell
def _(df_avg_odds, param_grid_weighted_nll):
    search_avg_odds_predictive_weighted_extreme = run_predictive_grid_search(
        model_factory=model_factory,
        param_grid=param_grid_weighted_nll,
        df=df_avg_odds,
        cache_mode="use",
    )
    return (search_avg_odds_predictive_weighted_extreme,)


@app.cell(hide_code=True)
def _():
    mo.md("""
    ### Średnie punktowanie na dodatkowej siatce weighted NLL

    Na tej samej dodatkowej siatce, na której wcześniej oceniany był weighted NLL, sprawdzamy również średnie punktowanie `avg_points`. To pozwala zestawić obszar preferowany przez weighted NLL z praktycznym celem konkursowym, analogicznie do wcześniejszego porównania siatki punktowej z metryką NLL. Dzięki temu widać, jak zmienia się średnia punktów w zakresie parametrów wybranym pod weighted NLL.
    """)
    return


@app.cell
def _(search_avg_odds_predictive_weighted_extreme):
    _ax = plot_grid_search_2d(
        search_avg_odds_predictive_weighted_extreme.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="avg_points",
    )
    _fig = plt.gcf()
    _fig.set_size_inches(12, 6)
    mo.output.append(_ax)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja:** Na tej dodatkowej siatce średnie punktowanie nie jest już tak chaotyczne jak na pierwotnej siatce, ale nadal optimum weighted NLL nie przekłada się na szczególnie dobry wynik konkursowy. Parametry wskazane przez weighted NLL dają raczej przeciętne średnie punktowanie, zwłaszcza w porównaniu z najlepszymi wartościami `avg_points` obserwowanymi na pierwszej siatce. To sugeruje, że weighted NLL w tym zakresie mocno odchodzi od celu konkursowego, mimo że poprawia własną metrykę probabilistyczną.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Odcięte średnie kursy (tylko NLL)

    Tu wejściem są **`trimmed_avg_*`** — średnia po bukmacherach po odrzuceniu skrajnych kursów (jak w opisie metodologii agregacji). **Dlaczego tylko `NLL`:** pełny blok z **`avg_points`**, **weighted NLL** i dodatkową siatką jest już rozpracowany na **średnich kursach**; powierzchnia **`avg_points`** przy innych agregacjach jest podobnie **niestabilna**, więc nie duplikujemy całego kosztownego rozdziału. Zamiast tego sprawdzamy jedno **porównywalne** kryterium probabilistyczne — **`NLL`** na **tej samej `param_grid`** — żeby zobaczyć, czy **inne ważenie bukmacherów** zmienia optimum `(ρ, bias)` i kształt heatmapy względem zwykłych średnich.
    """)
    return


@app.cell
def _(df_grid_search):
    df_avg_trimmed_odds = add_power_implied_probabilities_standard_markets(df_grid_search, odds_prefix='trimmed_avg')
    return (df_avg_trimmed_odds,)


@app.cell
def _(df_avg_trimmed_odds, param_grid):
    search_avg_trimmed_odds = run_predictive_nll_grid_search(
        model_factory=model_factory,
        param_grid=param_grid,
        df=df_avg_trimmed_odds,
        cache_mode="use"
    ) 
    return (search_avg_trimmed_odds,)


@app.cell
def _(search_avg_trimmed_odds):
    ax_avg_trimmed_odds = plot_nll_grid_search_2d(
        search_avg_trimmed_odds.results_df,
        x_param="bias_correction",
        y_param="rho",
    )

    # # Odświeżenie widoku 
    # # Zmiana rozmiaru: (szerokość, wysokość) w calach
    _fig = plt.gcf()
    _fig.set_size_inches(12, 6) 

    mo.output.append(ax_avg_trimmed_odds)
    return


@app.cell
def _(search_avg_trimmed_odds):
    mo.md(f"""
    Najlepsze parametry wybrane na podstawie Grid Search NLL dla średnich kursów obciętych:
    - rho: {np.round(search_avg_trimmed_odds.best_params['rho'], 2)}
    - bias_correction: {np.round(search_avg_trimmed_odds.best_params['bias_correction'], 2)}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja:** Heatmapa dla odciętych średnich kursów wygląda bardzo podobnie jak wcześniejsza heatmapa NLL dla zwykłych średnich kursów. Optimum znajduje się w tym samym obszarze, czyli w okolicy `rho = 0` oraz `bias_correction = 1.0`. To sugeruje, że usunięcie skrajnych kursów nie zmienia istotnie kalibracji modelu mierzonej przez NLL.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Maksymalne kursy (tylko NLL)

    Wejściem są **`max_*`** — dla każdego znaku rynku brany jest **najwyższy** kurs w zestawie bukmacherów (mniej marży „wbudowanej’’ w pojedynczą linię niż przy średniej). **Ta sama logika co przy odciętych średnich:** reprodukujemy wyłącznie **`NLL`** na **identycznej `param_grid`**, żeby oszacować wpływ agregacji na **optimum `(ρ, bias)` pod kątem NLL** bez ponownego uruchamiania całego łańcucha **`avg_points`** / weighted NLL. Oczekiwanie jest zbieżne z interpretacją pod heatmapą: jeśli optimum i kształt są bliskie wariantowi ze średnich, **sposób agregacji kursów ma tu drugorzędne znaczenie** przy tym kryterium.
    """)
    return


@app.cell
def _(df_grid_search):
    df_max_odds = add_power_implied_probabilities_standard_markets(df_grid_search, odds_prefix='max')
    return (df_max_odds,)


@app.cell
def _(df_max_odds, param_grid):
    search_max_odds = run_predictive_nll_grid_search(
        model_factory=model_factory,
        param_grid=param_grid,
        df=df_max_odds,
        cache_mode="use"
    ) 
    return (search_max_odds,)


@app.cell
def _(search_max_odds):
    ax_max_odds = plot_nll_grid_search_2d(
        search_max_odds.results_df,
        x_param="bias_correction",
        y_param="rho",
    )

    # Odświeżenie widoku 
    # Zmiana rozmiaru: (szerokość, wysokość) w calach
    _fig = plt.gcf()
    _fig.set_size_inches(12, 6) 

    mo.output.append(ax_max_odds)
    return


@app.cell
def _(search_max_odds):
    mo.md(f"""
    Najlepsze parametry wybrane na podstawie Grid Search NLL dla maksymalnych kursów:
    - rho: {np.round(search_max_odds.best_params['rho'], 2)}
    - bias_correction: {np.round(search_max_odds.best_params['bias_correction'], 2)}
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja:** Heatmapa dla maksymalnych kursów również ma bardzo podobny kształt do NLL dla średnich kursów, a optimum wypada w tym samym miejscu. Najlepsze NLL jest lepsze mniej więcej o jedną tysięczną, ale ta różnica jest zbyt mała, żeby traktować ją jako praktycznie istotną. Wniosek jest więc taki, że zmiana agregacji na maksymalne kursy nie prowadzi do wyraźnie innego optimum ani do znaczącej poprawy kalibracji.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    ## Wnioski z grid search

    Grid search nie okazał się wystarczająco dobrym narzędziem do jednoznacznego wyboru parametrów dla żadnej z analizowanych metryk. Metryka średniego punktowania (`avg_points`) jest zbyt niestabilna: niewielkie przesunięcie parametrów potrafi mocno zmienić wynik, więc trudno traktować pojedyncze optimum jako wiarygodny wybór modelu.

    Z kolei obie metryki NLL wskazały obszary, które nie są w pełni satysfakcjonujące z perspektywy celu konkursowego. Zwykłe NLL prowadzi do stabilnego, ale przeciętnie punktującego rozwiązania w okolicy `rho = 0` i `bias_correction = 1.0`. Weighted NLL próbuje lepiej uwzględnić logikę punktacji `3/2/1`, ale jego optimum wypada przy parametrach bardzo odległych od klasycznej interpretacji modelu: dodatnim `rho` i silnym zmniejszeniu bazowych lambd.

    Nie oznacza to jednak, że model należy odrzucić. Bazowe lambdy odtwarzane z kursów mogły być źle skalibrowane w sposób, którego jeden globalny mnożnik `bias_correction` nie jest w stanie naprawić. Z tego powodu kolejnym krokiem jest diagnostyka PIT, która pozwala ocenić kalibrację całych rozkładów prawdopodobieństwa, a nie tylko wartości metryk optymalizowanych na siatce parametrów.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # PIT

    PIT (*probability integral transform*) sprawdza, czy rozkład predykcyjny jest dobrze skalibrowany. Dla dyskretnych wyników bramkowych używamy losowanego PIT:

    $$U = F(y - 1) + V \cdot P(Y = y), \quad V \sim Uniform(0, 1)$$

    Jeśli model jest skalibrowany, histogramy PIT powinny być możliwie płaskie. Dodatkowo zgodność empirycznego rozkładu wartości PIT z rozkładem referencyjnym dla poprawnej kalibracji (w przybliżeniu jednostajnym na odcinku $[0, 1]$) sprawdzamy wykresami typu **QQ-worm** (worm plot): działają jak wizualny odpowiednik wykresu kwantyl-kwantyl i pokazują systematyczne odchylenia od oczekiwanego kształtu.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Diagnostyka kalibracji: PIT dla najlepszych parametrów

    Poniżej porównujemy trzy modele ze średnich kursów: najlepszy według średniej liczby punktów, najlepszy według NLL oraz najlepszy według ważonego NLL.
    """)
    return


@app.cell
def _():
    pit_variants = (
        "home_goals",
        "away_goals",
        "total_goals",
        "goal_difference",
    )
    pit_variant_groups = {
        "Home goals calibration": ("home_goals", "home_given_away"),
        "Totals and difference calibration": ("total_goals", "goal_difference"),
    }
    pit_random_states = np.arange(10_000, 10_100)
    return pit_random_states, pit_variants


@app.cell
def _(
    df_avg_odds,
    pit_random_states,
    pit_variants,
    search_avg_odds,
    search_avg_odds_nll,
    search_avg_odds_weighted_nll_extreme_grid,
):
    best_avg_points_model = model_factory(**search_avg_odds.best_params)
    best_avg_nll_model = model_factory(**search_avg_odds_nll.best_params)
    best_avg_weighted_nll_model = model_factory(**search_avg_odds_weighted_nll_extreme_grid.best_params)

    pit_avg_points_pred = best_avg_points_model.predict(df_avg_odds)
    pit_avg_nll_pred = best_avg_nll_model.predict(df_avg_odds)
    pit_avg_weighted_nll_pred = best_avg_weighted_nll_model.predict(df_avg_odds)

    pit_avg_points_result = build_pit_diagnostics(
        lambda_home=pit_avg_points_pred["exp_goals_home"],
        lambda_away=pit_avg_points_pred["exp_goals_away"],
        actual_home=pit_avg_points_pred["home_score"],
        actual_away=pit_avg_points_pred["away_score"],
        matrix_builder=best_avg_points_model.matrix_builder,
        variants=pit_variants,
        random_states=pit_random_states,
        model_name="best_avg_points",
        sample_name="grid_search_history",
    )
    pit_avg_nll_result = build_pit_diagnostics(
        lambda_home=pit_avg_nll_pred["exp_goals_home"],
        lambda_away=pit_avg_nll_pred["exp_goals_away"],
        actual_home=pit_avg_nll_pred["home_score"],
        actual_away=pit_avg_nll_pred["away_score"],
        matrix_builder=best_avg_nll_model.matrix_builder,
        variants=pit_variants,
        random_states=pit_random_states,
        model_name="best_avg_nll",
        sample_name="grid_search_history",
    )
    pit_avg_weighted_nll_result = build_pit_diagnostics(

        lambda_home=pit_avg_weighted_nll_pred["exp_goals_home"],
        lambda_away=pit_avg_weighted_nll_pred["exp_goals_away"],
        actual_home=pit_avg_weighted_nll_pred["home_score"],
        actual_away=pit_avg_weighted_nll_pred["away_score"],
        matrix_builder=best_avg_weighted_nll_model.matrix_builder,
        variants=pit_variants,
        random_states=pit_random_states,
        model_name="best_avg_weighted_nll",
        sample_name="grid_search_history",
    )
    return (
        pit_avg_nll_result,
        pit_avg_points_result,
        pit_avg_weighted_nll_result,
    )


@app.cell
def _(pit_avg_nll_result, pit_avg_points_result, pit_avg_weighted_nll_result):
    pit_fig_history_home = plot_pit_histogram_replicates(
        {
            "Best by avg_points": pit_avg_points_result,
            "Best by avg_nll": pit_avg_nll_result,
            "Best by avg_weighted_nll": pit_avg_weighted_nll_result,
        },
        title="Repeated randomized PIT histograms - avg odds grid-search sample",
        figsize=(14,7)
    )
    mo.output.append(pit_fig_history_home)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja PIT - histogram:** Histogramy są w przybliżeniu równomierne dla parametrów wybranych według metryki <span style='color: #2563eb;'>średnich punktów</span> (`avg_points`) oraz dla parametrów z optymalnego <span style='color: #ea580c;'>średniego NLL</span> — na tym poziomie trudno wychwycić drobne odchylenia, dlatego sensowniej je ocenić na następnym wykresie typu QQ-worm dla tych samych wartości PIT.

    Dla parametrów z optimum <span style='color: #16a34a;'>weighted NLL</span> widać już wyraźny problem przy predykcjach liczby goli gospodarzy i gości: wartości PIT najczęściej skupiają się blisko `1`. Oznacza to, że obserwowane wyniki leżą systematycznie w górnej części rozkładu predykcyjnego, czyli model przewiduje za mało bramek i rozkład jest w tych wariantach źle skalibrowany względem rzeczywistości.
    """)
    return


@app.cell
def _(pit_avg_nll_result, pit_avg_points_result, pit_avg_weighted_nll_result):
    pit_worm_fig_history_home = plot_pit_worm_replicates(
        {
            "Best by avg_points": pit_avg_points_result,
            "Best by avg_nll": pit_avg_nll_result,
            "Best by avg_weighted_nll": pit_avg_weighted_nll_result,
        },
        title="Repeated randomized PIT worm plot - Home goals calibration - avg odds grid-search sample",
        figsize=(14,8)
    )
    mo.output.append(pit_worm_fig_history_home)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja PIT — worm plot (QQ-worm):**

    **Model z parametrami z optimum <span style='color: #ea580c;'>NLL</span>.** Dla goli gospodarzy kształt „robaka’’ jest zbliżony do oczekiwanego przy dobrej kalibracji. Dla goli gości widać odchylenie: rozkład predykcyjny koncentruje się zbyt mocno na zbyt małej liczbie bramek — krzywa leży w ponad połowie powtórzeń powyżej symulowanego **95% pasma** rozkładu referencyjnego (jednostajnego). Suma goli wygląda rozsądnie, natomiast różnica bramek schodzi nieco **pod** to pasmo; taki obraz może w dużej mierze wynikać z wcześniejszej, gorszej kalibracji strony wyjazdowej.

    **Model z parametrami z optimum <span style='color: #2563eb;'>średniej punktowej</span> (`avg_points`).** Dla goli gospodarzy worm plot schodzi **poniżej** pasma referencyjnego, co jest zgodne z **zawyżonym** prognozowanym strzelaniem przy optymalnym dla tej metryki `bias_correction` około `1.02` (dla optimum <span style='color: #ea580c;'>NLL</span> `bias_correction` był bliżej jedynki i tam dla goli domowych wyglądało stabilniej). Bramki wyjazdowe są nadal źle skalibrowane w podobnym kierunku co przy <span style='color: #ea580c;'>NLL</span>, ale krzywa **bliżej** znajduje się w pobliżu dopuszczalnego pasma — znów można to wiązać z rolą biasu przy budowie lambdy. Dodatkowo różnica goli oraz suma goli pokazują lokalne odchylenia (różnica goli wyraźniej, suma goli w jednym segmencie).

    **Syntetyczny wątek.** Wygląda na to, że przy lambdach bez silnej korekty `bias_correction` strona domowa może być przyzwoicie skalibrowana, ale wybrany pod punkty bias ją przestaje. Jednocześnie gole wyjazdowe są problematyczne w obu porównywanych wariantach, przy czym przy modelu pod <span style='color: #2563eb;'>średnim punktowaniem</span> (`avg_points`) są nieco „bliżej’’ pasma niż przy <span style='color: #ea580c;'>NLL</span>. To sugeruje, że **osobna, precyzyjniejsza korekta lambdy wyjazdowej** mogłaby być naturalnym kolejnym krokiem w duchu dopasowania biasu nie tylko globalnie dla obu drużyn naraz.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## PIT dla metod agregacji kursów przy najlepszych parametrach NLL

    Porównujemy kalibrację PIT dla trzech metod agregacji kursów:
    średnich, średnich obciętych i maksymalnych. Parametry modelu są stałe:
    `rho=0`, `bias_correction=1.0`, ponieważ taki układ okazał się najlepszy
    dla średniego NLL z grid searchy metod agregacji.
    """)
    return


@app.cell
def _(
    df_avg_odds,
    df_avg_trimmed_odds,
    df_max_odds,
    pit_random_states,
    pit_variants,
):
    pit_aggregation_nll_model = model_factory(rho=0, bias_correction=1.0)
    pit_aggregation_inputs = {
        "Avg odds": df_avg_odds,
        "Trimmed avg odds": df_avg_trimmed_odds,
        "Max odds": df_max_odds,
    }

    pit_aggregation_results = {}
    for aggregation_name, aggregation_df in pit_aggregation_inputs.items():
        _pred = pit_aggregation_nll_model.predict(aggregation_df)
        pit_aggregation_results[aggregation_name] = build_pit_diagnostics(
            lambda_home=_pred["exp_goals_home"],
            lambda_away=_pred["exp_goals_away"],
            actual_home=_pred["home_score"],
            actual_away=_pred["away_score"],
            matrix_builder=pit_aggregation_nll_model.matrix_builder,
            variants=pit_variants,
            random_states=pit_random_states,
            model_name=aggregation_name,
            sample_name="grid_search_history_fixed_nll_params",
        )
    return (pit_aggregation_results,)


@app.cell
def _(pit_aggregation_results):
    pit_aggregation_hist_fig = plot_pit_histogram_replicates(
        pit_aggregation_results,
        title="Repeated randomized PIT histograms - aggregation methods, rho=0, bias=1.0",
        figsize=(14, 7),
    )
    mo.output.append(pit_aggregation_hist_fig)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja PIT agregacji — histogram:** Przy stałych parametrach `rho=0` oraz `bias_correction=1.0` histogramy powtórzonego PIT dla średnich, obciętych średnich i maksymalnych kursów są do siebie bardzo zbliżone. Nie widać wyraźnych, systematycznych różnic między metodami agregacji — żadna z nich nie prowadzi wyraźnie do innej kalibracji w tym ujęciu.
    """)
    return


@app.cell
def _(pit_aggregation_results):
    pit_aggregation_worm_fig = plot_pit_worm_replicates(
        pit_aggregation_results,
        title="Repeated randomized PIT worm plot - aggregation methods, rho=0, bias=1.0",
        figsize=(14, 8),
    )
    mo.output.append(pit_aggregation_worm_fig)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja PIT agregacji — worm plot:** Na wykresach typu QQ-worm sytuacja jest spójna z histogramami: krzywe dla średnich, obciętych średnich i maksymalnych kursów przebiegają bardzo podobnie. Różnice między typami agregacji nie są w praktyce istotne przy tym zestawie parametrów.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Test out-of-sample (aktualny sezon)

    Parametry modelu (`ρ`, `bias_correction`) dobieramy na wielosezonowej próbie historycznej; ta sekcja sprawdza, jak modele z tym doborem zachowują się na **świeższej próbie** — meczach aktualnego sezonu od startu konkursu Supertyper (jak wyżej: `season == "current"` od ustalonej daty).

    To nie jest pełna procedura walidacji czasowej (np. walk-forward po kolejnych kolejkach), lecz **praktyczny test porównawczy**: czy kalibracja i jakość predykcji przypominają to, co widzieliśmy na historii.

    Na heatmapach poniżej metryka to **`total_points`** — suma punktów Supertypera na rozegranych meczach z tej próby. Dla skali konkursowej w tej rundzie przyjmijmy orientacyjnie: około **154 pkt** odpowiada poziomowi **lidera**, a około **130 pkt** to wynik **bardzo przeciętny** w tabeli.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Powierzchnia punktacji na próbie konkursowej

    Nie wykonujemy tu ponownego „szukania” modelu od zera na próbie konkursowej: **punktowo przeliczamy te same kombinacje** `(bias_correction, ρ)`, które badaliśmy wcześniej na historii (oraz osobno szerszą siatkę z sekcji weighted NLL), i patrzymy na **powierzchnię sumy punktów** przy tej samej logice wyboru typu co w `run_predictive_grid_search`.

    Zmiana zbioru meczów (mniejsza próba, inny terminarz i kontekst rynku) może przesunąć optymalny punkt siatki względem historii — to oczekiwane. Skala wartości na osi kolorów odnosi się do **sumy punktów** na tej próbie; orientacyjnie **~154 pkt** to poziom lidera, **~130 pkt** bardzo przeciętny wynik w konkursie.
    """)
    return


@app.cell
def _(df_current_season):
    # wiem że tutaj nie ma braków danych
    df_avg_odds_current = add_power_implied_probabilities_standard_markets(df_current_season, odds_prefix='avg')
    return (df_avg_odds_current,)


@app.cell
def _(df_avg_odds_current, param_grid):
    search_avg_odds_current = run_predictive_grid_search(
        model_factory=model_factory,
        param_grid=param_grid,
        df=df_avg_odds_current,
        score_key="total_points",
        cache_mode="use"
    ) 
    return (search_avg_odds_current,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Ta sama siatka co na historii (bias × ρ)

    Ta heatmapa używa **identycznego zakresu** parametrów co główny grid search na próbie historycznej. Pokazuje, jak układa się **suma punktów Supertypera** na próbie konkursowej, gdyby „sunąć” po tej samej siatce co wcześniej.
    """)
    return


@app.cell
def _(search_avg_odds_current):
    ax_avg_odds_current = plot_grid_search_2d(
        search_avg_odds_current.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="total_points"
    )

    # Odświeżenie widoku 
    mo.output.append(ax_avg_odds_current)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacja — ta sama siatka co na historii:**

    Na tej próbie konkursowej warto zestawić sumę punktów (`total_points`) w dwóch punktach wybranych wcześniej na podstawie historii grid search:

    - Kombinacja z optimum **NLL** (`rho = 0.00`, `bias_correction = 1.00`) daje tu około **140 pkt**. Przy skali z wstępu do sekcji (lider ok. **154 pkt**, bardzo przeciętnie ok. **130 pkt**) jest to **wynik środka tabeli**. Na tej samej siatce wiele innych par `(ρ, bias)` wygląda podobnie lub lepiej, więc **jest duże prawdopodobieństwo**, że losowy inny zestaw parametrów z siatki dałby nie gorszy — albo lepszy — wynik niż świadomy wybór pod kątem NLL.

    - Kombinacja z optimum **`avg_points`** (`rho = -0.18`, `bias_correction = 1.02`) na historii osiąga około **148 pkt**, czyli wyraźnie lepiej niż punkt NLL i zbliża się do sensownego poziomu konkursowego. Jednocześnie powierzchnia jest **bardzo niestabilna**: przy **ρ** niższym o **0.02** od optimum pod średnią punktację suma schodzi już do około **140 pkt**, czyli z powrotem do strefy przeciętnej. To podkreśla, że optimum `avg_points` na krótkiej próbie nie jest ostrym ani stabilnym szczytem powierzchni — wokół niego leży sąsiedztwo o zbliżonym lub znacznie gorszym wyniku.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Rozszerzona siatka (jak przy weighted NLL)

    Ta sama **próba konkursowa**, ale punkty `(bias_correction, ρ)` z **przesuniętego zakresu** użytego przy dodatkowej siatce dla weighted NLL na historii. Chodzi o to, czy na OOS widać podobne zachowanie (np. czy ekstremalne wartości parametrów nadal dominują sumę punktów) oraz jak wygląda powierzchnia w obszarze, który na historii był istotny dla weighted NLL.
    """)
    return


@app.cell
def _(df_avg_odds_current, param_grid_weighted_nll):
    search_avg_odds_currentgrid_weighted_nll = run_predictive_grid_search(
        model_factory=model_factory,
        param_grid=param_grid_weighted_nll,
        df=df_avg_odds_current,
        score_key="total_points",
        cache_mode="use"
    ) 
    return (search_avg_odds_currentgrid_weighted_nll,)


@app.cell
def _(search_avg_odds_currentgrid_weighted_nll):
    _ax = plot_grid_search_2d(
        search_avg_odds_currentgrid_weighted_nll.results_df,
        x_param="bias_correction",
        y_param="rho",
        metric_name="total_points"
    )

    # Odświeżenie widoku 
    mo.output.append(_ax)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacja — rozszerzona siatka (jak przy weighted NLL):**

    Obszar parametrów widoczny na tej heatmapie jest dla próby konkursowej **jednostajnie słaby**. Większość widocznych kombinacji skupia się w okolicach około **130 pkt**, czyli przy przyjętej skali odpowiada **bardzo przeciętnemu** poziomowi w tabeli. Nieliczne komórki sięgają około **140 pkt**, ale to nadal **słabe** w zestawieniu z poziomem lidera (~**154**) i nie rekompensuje ekstremalnych wartości `ρ` oraz `bias_correction`, które na historii uzasadniały poszukiwanie optimum weighted NLL. Obraz jest spójny z wcześniejszą diagnozą: **region preferowany przez weighted NLL na historii nie przekłada się na atrakcyjną sumę punktów OOS** na tej próbie.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## PIT out-of-sample na aktualnym sezonie

    Te same **trzy** warianty parametrów co na historii grid search — **najlepsze według `avg_points`**, **najlepsze według NLL** oraz **najlepsze według weighted NLL** (siatka rozszerzona) — sprawdzamy na `df_avg_odds_current`. To jest ważniejsze diagnostycznie niż PIT wyłącznie na wielosezonowej próbie użytej do siatki parametrów, bo pokazuje kalibrację **poza** historią wyboru hiperparametrów.

    Próba konkursowa jest **krótsza** niż historia grid search, więc losowość powtórzonego PIT i pasmo symulacji na worm plotach będzie **szersze** — drobne różnice między modelami traktuj ostrożniej niż na dużej próbie.
    """)
    return


@app.cell
def _(
    df_avg_odds_current,
    pit_random_states,
    pit_variants,
    search_avg_odds,
    search_avg_odds_nll,
    search_avg_odds_weighted_nll_extreme_grid,
):
    best_avg_points_model_current_pit = model_factory(
        **search_avg_odds.best_params
    )
    best_avg_nll_model_current_pit = model_factory(
        **search_avg_odds_nll.best_params
    )
    best_avg_weighted_nll_model_current_pit = model_factory(
        **search_avg_odds_weighted_nll_extreme_grid.best_params
    )

    pit_current_avg_points_pred = best_avg_points_model_current_pit.predict(
        df_avg_odds_current
    )
    pit_current_avg_nll_pred = best_avg_nll_model_current_pit.predict(
        df_avg_odds_current
    )
    pit_current_avg_weighted_nll_pred = best_avg_weighted_nll_model_current_pit.predict(
        df_avg_odds_current
    )

    pit_current_avg_points_result = build_pit_diagnostics(
        lambda_home=pit_current_avg_points_pred["exp_goals_home"],
        lambda_away=pit_current_avg_points_pred["exp_goals_away"],
        actual_home=pit_current_avg_points_pred["home_score"],
        actual_away=pit_current_avg_points_pred["away_score"],
        matrix_builder=best_avg_points_model_current_pit.matrix_builder,
        variants=pit_variants,
        random_states=pit_random_states,
        model_name="best_avg_points",
        sample_name="current_season_oos",
    )
    pit_current_avg_nll_result = build_pit_diagnostics(
        lambda_home=pit_current_avg_nll_pred["exp_goals_home"],
        lambda_away=pit_current_avg_nll_pred["exp_goals_away"],
        actual_home=pit_current_avg_nll_pred["home_score"],
        actual_away=pit_current_avg_nll_pred["away_score"],
        matrix_builder=best_avg_nll_model_current_pit.matrix_builder,
        variants=pit_variants,
        random_states=pit_random_states,
        model_name="best_avg_nll",
        sample_name="current_season_oos",
    )
    pit_current_avg_weighted_nll_result = build_pit_diagnostics(
        lambda_home=pit_current_avg_weighted_nll_pred["exp_goals_home"],
        lambda_away=pit_current_avg_weighted_nll_pred["exp_goals_away"],
        actual_home=pit_current_avg_weighted_nll_pred["home_score"],
        actual_away=pit_current_avg_weighted_nll_pred["away_score"],
        matrix_builder=best_avg_weighted_nll_model_current_pit.matrix_builder,
        variants=pit_variants,
        random_states=pit_random_states,
        model_name="best_avg_weighted_nll",
        sample_name="current_season_oos",
    )
    return (
        pit_current_avg_nll_result,
        pit_current_avg_points_result,
        pit_current_avg_weighted_nll_result,
    )


@app.cell
def _(
    pit_current_avg_nll_result,
    pit_current_avg_points_result,
    pit_current_avg_weighted_nll_result,
):
    pit_current_fig = plot_pit_histogram_replicates(
        {
            "Best by avg_points": pit_current_avg_points_result,
            "Best by avg_nll": pit_current_avg_nll_result,
            "Best by avg_weighted_nll": pit_current_avg_weighted_nll_result,
        },
        title="Repeated randomized PIT histograms - current season out-of-sample",
        figsize=(14,7)
    )
    mo.output.append(pit_current_fig)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja PIT out-of-sample — histogram:**

    Próba konkursowa jest **wyraźnie mniejsza** niż historia użyta wcześniej. Dla powtórzonego, randomizowanego PIT oznacza to **szerszą niepewność odczytu** tego, co można uznać za „prawie płaski’’ histogram przy poprawnej kalibracji — nawet rozkład blisko jednostajnego na $[0, 1]$ na małej próbie może wyglądać nierówno. Dlatego dla wariantów **best by `avg_points`** oraz **best by NLL** fakt, że histogramy **nie są idealnie równe**, **nie pozwala sam z siebie** twierdzić o złej kalibracji: różnice mogą być po prostu **szumem małego zbioru**.

    Wyjątkiem jest **best by weighted NLL**: **mimo** ostrożnego czytania małej próby widać **ten sam obraz co na próbie historycznej**, do której model był dopasowywany — **większość wartości PIT skupia się przy wartościach bliskich 1** (prawa strona odcinka $[0, 1]$). To nie wygląda na przypadkowe wybrzuszenie przy krótkiej próbie: sygnał jest **systematyczny** i zgodny z wcześniejszą diagnozą (realizacje leżą systematycznie **wysoko** w rozkładzie predykcyjnym liczby goli — model **podwykonuje** względem obserwowanych wyników).
    """)
    return


@app.cell
def _(
    pit_current_avg_nll_result,
    pit_current_avg_points_result,
    pit_current_avg_weighted_nll_result,
):
    pit_current_worm_fig = plot_pit_worm_replicates(
        {
            "Best by avg_points": pit_current_avg_points_result,
            "Best by avg_nll": pit_current_avg_nll_result,
            "Best by avg_weighted_nll": pit_current_avg_weighted_nll_result,
        },
        title="Repeated randomized PIT worm plot - current season out-of-sample",
        figsize=(14,8)
    )
    mo.output.append(pit_current_worm_fig)
    return


@app.cell(hide_code=True)
def _():
    mo.md("""
    **Interpretacja PIT out-of-sample — worm plot:**

    Wykresy **potwierdzają to, co sugerują histogramy**: dla dwóch pierwszych modeli odchylenia od pasma referencyjnego da się często **wiązać** ze **szerszą niepewnością** symulacji przy krótszej próbie — pasma są tu **szersze** niż na dużej historii, więc pojedyncze wyjście krzywej poza nie nie jest automatycznie dowodem złej kalibracji.

    W **porównaniu** wygląda tak, że dla **best by NLL** — w przeciwieństwie do **best by `avg_points`** — na wariantach **`total_goals`** oraz **`away_goals`** widać **fragment**, w którym „robak’’ wychodzi **nieco poza** symulowane pasmo. Może to sugerować gorszą kalibrację tych składowych przy parametrach z optimum NLL, ale przy **tylko ok. 163 meczach** w próbie konkursowej taki **lokalny** wykrok poza przedział **równie dobrze** może wynikać z **losowości krótkiej próby** niż z trwałego, systematycznego błędu modelu — tu warto traktować sygnał **ostrożniej** niż w przypadku weighted NLL poniżej.

    Jednocześnie dla **best by weighted NLL** worm ploty **potwierdzają histogramy** i są **spójne z próbą dopasowania**: nadal dominują wartości PIT **blisko 1**, a krzywe odbiegają od pasma referencyjnego w sposób **systematyczny**, nie „rozmyty’’ jak przy samym szumie małej próby.

    Podsumowując: **mniejszy zbiór = ostrożniejsze winiowanie `avg_points` / NLL**, natomiast **weighted NLL na OOS nadal wygląda na źle skalibrowane rozkłady**, tak jak przy ocenie na danych, na których wybierano parametry.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Szczegółowy zrzut predykcji

    Poniżej **lista mecz po meczu** na **`df_avg_odds_current`** — tej samej próbie konkursowej co wyżej na wykresach OOS. To nie osobny eksperyment statystyczny, tylko **czytelny zrzut** tego, co model typuje przy dwóch **konkretnych zestawach** `(bias_correction, ρ)` z wcześniejszego grid search.

    Pokazujemy świadomie **kontrast krańcowy** między wariantami porównywanymi w raporcie pod **`avg_points`**, **NLL** i **weighted NLL** na historii:

    - najpierw parametry najlepszego z nich przy **celu konkursowym** — optimum **`avg_points`** (punkty konkursowe na długiej próbie wg tego notatnika);
    - niżej parametry wyboru **weighted NLL** na rozszerzonej siatce, które w dyskusji nad OOS/PIT oraz sumą **`total_points`** wypadły **decydowanie najsłabiej**, więc traktujemy je tu jako **najgorszy z trzech dla praktycznego celu konkursowego**.

    **`bias_correction`** i **`ρ`** jak w **`search_avg_odds.best_params`** oraz w **`search_avg_odds_weighted_nll_extreme_grid.best_params`**.

    **Znaczenie części kolumn:**
    - **`pred_score`** — wybrany typ dokładnego wyniku (`gospodarz:gość`);
    - **`points_score`** — punkty zdobyte przy znanym wyniku (0/1/2/3 według zasad Supertyper);
    - **`pred_xpts`** — oczekiwana liczba punktów konkursowych z dystrybucji predykcyjnej przed meczem;
    - **`exp_goals_home`** / **`exp_goals_away`** — oczekiwane gole po modelu (Poisson–Dixon–Coles na lambdach z kursów × `bias_correction`);
    - **`avg_1`**, **`avg_X`**, **`avg_2`** — zagregowane średnie kursy (`1`, remis, `2`) dla kontekstu.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Wariant najlepszy spośród kryteriów (optymum `avg_points` na historii)

    **`bias_correction = 1.02`**, **`ρ = −0.18`** — konkretyzacje optimum pod **średnią punktację konkursową** na próbie multi‑sezonowej z wcześniejszego grid search (**średnie kursy**). To zestaw przyklejamy do próby konkursowej jako **„go‑to’’** z perspektywy tego raportu przy maksymalizacji punków zgodnych z konkursiem.
    """)
    return


@app.cell
def _():
    potential_best_model = PoissonDixonColesModel(bias_correction=1.02, rho=-0.18)
    return (potential_best_model,)


@app.cell
def _(df_avg_odds_current, potential_best_model):
    df_potential_best = potential_best_model.predict(df_avg_odds_current)
    return (df_potential_best,)


@app.cell
def _(df_potential_best):
    df_potential_best['points_score'] = compute_points_per_match(df_potential_best)
    return


@app.cell
def _(df_potential_best):
    columns_to_show = [
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
    df_potential_best[columns_to_show]
    return (columns_to_show,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Wizualne podsumowanie predykcji (próba konkursowa)

    Poniżej **dwa** kolejne wykresy na tej samej próbie co powyższa tabela (`df_potential_best`).

    - **`plot_predictions_summary`** — histogram punktów Supertypera, podsumowanie **sumy / średniej** oraz **macierz pomyłek 1X2** (typ modelu vs rzeczywisty wynik meczowy `1` / `X` / `2`).

    - **`plot_predictions_scoreline_summary`** — **dokładny wynik bramkowy**: górny rząd to dwie heatmapy częstości par *(gole gospodarza, gole gości)* przy **ścięciu** wysokich wartości do ostatniego kubła na osiach (przy domyślnych ustawieniach typowo etykiety **0 … +4** na siatce); dolny rząd to **najczęstsze** zapisy `h:a` dla predykcji i dla rzeczywistych wyników (top‑N; słupki oparte na pełnych liczbach całkowitych, bez klipowania).
    """)
    return


@app.cell
def _(df_potential_best):
    plot_predictions_summary(df_potential_best, model_name="Poisson Dixon-Coles (baseline)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(df_potential_best):
    plot_predictions_scoreline_summary(df_potential_best, model_name="Rozkłady wyników - Poisson Dixon-Coles (baseline)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacja (wariant optimum `avg_points` na historii):**

    Na próbie konkursowej (**163 mecze**) model zbiera **148 pkt** (**~0,908 pkt/mecz**).

    **Macierz 1X2** pokazuje, że przy **`1`** i **`2`** model radzi sobie **wyraźnie lepiej** niż przy **`X`**: poprawnych rozstrzygnięć typu zwycięstwo gospodarza lub gościa jest **znacznie więcej** niż poprawnych **remisów**. **Remisy są najsłabszym segmentem** — model często zamiast **`X`** typuje **`1`** lub **`2`**, więc pod kątem samego trafienia 1X2 **nie jest tak skuteczny na remisy** jak na zwycięstwa którejś ze stron.

    **Heatmapy par bramek** oraz **top‑N `h:a`** pokazują, że **zestaw najczęstszych typowanych wyników jest zbliżony do zestawu najczęstszych faktycznych** (te same niskie rezultaty typu **2:1**, **1:2**, **1:1**, **2:0** dominują po obu stronach). Jednocześnie **w realizacji częściej padał remis** niż sugeruje sam układ predykcji — faktycznie najczęstszy mecz to **1:1**, podczas gdy typy są **silniej rozłożone na scenariusze ze zwycięzcą**, co wiąże się z omówioną wyżej **trudnością z `X`** na macierzy 1X2.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Wariant najsłabszy przy celu konkursowym (optymum weighted NLL na historii)

    **`bias_correction = 0.46`**, **`ρ = 0.68`** — jak w **`search_avg_odds_weighted_nll_extreme_grid.best_params`** (rozszerzona siatka pod weighted NLL). W rozdziale OOS oraz PIT wariant ten traktujemy jak **„najgorzej’’** dla praktycznego konkursowego rezultatu wśród trzech kryteriów; zrzut tabelowy i wykresy niżej służą **porównaniu z punktem odniesienia** powyżej, **bez rekomendacji** użytku tych wartości w produkcji.
    """)
    return


@app.cell
def _():
    potential_best_weighted_nll = PoissonDixonColesModel(bias_correction=0.46, rho=0.68)
    return (potential_best_weighted_nll,)


@app.cell
def _(df_avg_odds_current, potential_best_weighted_nll):
    df_potential_best_weighted_nll = potential_best_weighted_nll.predict(df_avg_odds_current)
    return (df_potential_best_weighted_nll,)


@app.cell
def _(df_potential_best_weighted_nll):
    df_potential_best_weighted_nll['points_score'] = compute_points_per_match(df_potential_best_weighted_nll)
    return


@app.cell
def _(columns_to_show, df_potential_best_weighted_nll):
    df_potential_best_weighted_nll[columns_to_show]
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    #### Wizualne podsumowanie — weighted NLL (do porównania)

    Ta sama **kolejność** i **typy** figur co wyżej, na ramce `df_potential_best_weighted_nll`. Powtórnego opisu funkcji nie ma — chodzi o **porównanie kształtu** błędów z wariantem optimum **`avg_points`** przy tej samej próbie konkursowej.
    """)
    return


@app.cell
def _(df_potential_best_weighted_nll):
    plot_predictions_summary(df_potential_best_weighted_nll, model_name="Poisson Dixon-Coles (weighted NLL)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---
    """)
    return


@app.cell
def _(df_potential_best_weighted_nll):
    plot_predictions_scoreline_summary(df_potential_best_weighted_nll, model_name="Rozkłady wyników - Poisson Dixon-Coles (weighted NLL)")
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    **Interpretacja — oba wykresy (wariant optimum weighted NLL na historii):**

    W zestawieniu z optimum **`avg_points`** ten wariant na próbie konkursowej wygląda **wyraźnie gorzej** pod kątem sumy punktów — zbieżnie z wcześniejszą oceną OOS i powierzchniami **`total_points`**.

    Na heatmapach oraz w **top `h:a`** model przy tych parametrach **masowo typuje `1:0` i `0:1`**. Takie skupienie na minimalnych zwycięstwach **w ogóle słabo zgrywa się z realizacją**: przy rozstrzygnięciach ze **zwycięstwem gospodarza przy różnicy jednej bramki** **`1:0` nie jest najczęstszym dokładnym wynikiem** w próbie — empirycznie częstsza jest struktura wyższych „minimalnych’’ zwycięstw (**np. `2:1`**), podobnie po stronie gościa sensowniej wygląda **`1:2`** niż **`0:1`** jako masowy typ przy zwycięstwie przy jednej bramce.

    **Konkursowo** bardziej opłacałoby się więc **zastąpić** dominujące **`1:0` / `0:1`** typami **`2:1`** i **`1:2`**: przy **tym samym rozstrzygnięciu 1X2** lepsza jest szansa na punkty przy dokładnym wyniku zgodnym z **częstszym profilem faktycznych rezultatów** niż przy czystych jednobramkowych minimalnych zwycięstwach, których model tu nadmiernie namnaża.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Podsumowanie

    - **Model bazowy (Baseline)**
    Raport przedstawia model bazowy oparty na rozkładzie Poissona z poprawką Dixona–Colesa. Parametry $\lambda$ zostały odtworzone z kursów bukmacherskich za pomocą metody Power Implied Probabilities. Globalne parametry $\rho$ oraz bias_correction optymalizowano pod kątem punktacji konkursowej (system 3/2/1/0) oraz wybranych metryk probabilistycznych.

    - **Przeszukiwanie siatki (Grid Search)**
    Optymalizacja pod kątem średniej liczby punktów (avg_points) na długiej próbie historycznej prowadzi do wyższych wyników konkursowych niż w przypadku optymalizacji NLL (Negative Log-Likelihood). Niestety, powierzchnia funkcji celu dla avg_points jest niestabilna, co sprawia, że znalezienie jednego, pewnego optimum jest trudne. Z kolei metryka NLL charakteryzuje się znacznie gładszą powierzchnią (z optimum w okolicach $\rho \approx 0$ i bias_correction $\approx 1$), choć daje średnio niższe wyniki w samej punktacji konkursowej. Zastosowanie ważonego NLL (weighted NLL, w tym na rozszerzonej siatce) skutkuje ekstremalnymi wartościami parametrów, słabym wynikiem na próbie testowej (OOS) oraz wyraźnym zaburzeniem kalibracji (widocznym na wykresach PIT). Ten wariant należy traktować jako antywzorzec, a nie użyteczne rozwiązanie analityczne.

    - **Analiza PIT i próba OOS (Out-of-Sample)**
    Diagnostyka PIT oraz mapy ciepła dla total_points na próbie OOS są spójne z powyższymi wnioskami. Ważony NLL silnie zaburza rozkład prawdopodobieństw dla liczby goli (skupienie wartości PIT blisko 1). W przypadku avg_points oraz standardowego NLL, interpretacja wykresów typu worm plot na krótkiej próbie wymaga ostrożności ze względu na dużą niepewność estymacji. Analiza macierzy dokładnych wyników (home:away) ujawnia specyficzne profile błędów: optimum avg_points słabiej radzi sobie z identyfikacją remisów, natomiast optimum weighted NLL masowo i błędnie przewiduje wyniki 1:0 lub 0:1, ignorując naturalną częstotliwość rezultatów takich jak 2:1 czy 1:2 w analizowanej próbie.

    - **Wnioski z optymalizacji metryk (NLL vs Weighted NLL)**
    Słabsze wyniki punktowe przy optymalizacji pod standardowe NLL nie są wadą samej metryki, która z natury rygorystycznie ocenia kalibrację. Wynikają one z ograniczeń obecnego modelu, w którym optymalizowany był zaledwie jeden, globalny parametr dla lambd obu drużyn naraz. NLL ma jednak duży potencjał i będzie optymalnym wyborem w przyszłych, bardziej elastycznych modelach. Z kolei skrajnie złe wyniki i zaburzona kalibracja przy ważonym NLL (Weighted NLL) sugerują patologię samej funkcji kosztu. Optymalizator "oszukuje" system wagowy – poprzez drastyczne zaniżenie bazowych lambd sztucznie zwęża wariancję rozkładu, kompresując masę prawdopodobieństwa na najniższych możliwych wynikach (np. 1:0, 0:1), co pozwala matematycznie zmaksymalizować metrykę, ale całkowicie odrywa model od boiskowych realiów.

    - **Agregacja kursów bukmacherskich**
    Główna część raportu opiera się na kursach uśrednionych. Dla średnich obciętych (trimmed_avg) oraz wartości maksymalnych przeprowadzono dodatkową ewaluację NLL na tej samej siatce parametrów. Wyniki pokazują, że metoda agregacji ma marginalny wpływ na położenie optimum parametrów $(\rho, \text{bias})$. Kluczowym wyzwaniem pozostaje prawidłowe odtworzenie samych wartości $\lambda$. Z punktu widzenia praktyki projektowej zaleca się jednak stosowanie średniej obciętej. Procedura ta eliminuje wartości skrajne (np. błędy wystawienia linii lub specyfikę konkretnego bukmachera), oferując dodatkowe zabezpieczenie stabilności modelu bez istotnego wpływu na ogólną strukturę danych.


    - **Kierunki dalszych prac i weryfikacja na danych syntetycznych**
    Aby ostatecznie zweryfikować zachowanie metryki NLL oraz zdiagnozować patologię ważonego NLL, planowane jest przeprowadzenie eksperymentów na danych syntetycznych. Wygenerowanie wyników spotkań z ustaloną "prawdą podstawową" dla parametrów $\lambda$ i $\rho$ pozwoli sprawdzić, czy optymalizatory są w stanie bezbłędnie je odzyskać i matematycznie udowodni, czy ważone NLL wprowadza systematyczny błąd. Kolejnym kluczowym krokiem będzie odejście od jednego globalnego mnożnika (bias_correction). Konieczne jest osobne modelowanie $\lambda_{\text{home}}$ oraz $\lambda_{\text{away}}$ z wykorzystaniem bardziej złożonych metod (np. modeli GLM), w których NLL będzie mogło optymalizować większą liczbę parametrów. Jest to niezbędne, zanim uzna się za zamkniętą procedurę typowania opartą wyłącznie na kursach.


    - **Ograniczenia**
    Obecny model opiera się wyłącznie na dostępnych kursach bukmacherskich. Brakuje w nim szczegółowych statystyk meczowych oraz danych zdarzeniowych, które w naturalny sposób zwiększyłyby precyzję predykcji. Pozyskanie i integracja tych informacji to kolejny etap prac, który powinien nastąpić po ustabilizowaniu kalibracji parametrów $\lambda$ – tak, aby uniknąć konieczności opierania się na ekstremalnych wartościach parametrów Dixona-Colesa, wymuszanych przez nieodpowiednio dobrane funkcje kosztu.
    """)
    return


if __name__ == "__main__":
    app.run()
