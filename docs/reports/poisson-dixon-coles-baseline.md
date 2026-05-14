---
title: Raport baseline Poisson Dixon–Coles
summary: Przewodnik po raporcie HTML — zakres analizy, wnioski i kolejność czytania sekcji
sidebar_title: Raport Poisson DC baseline
order: 1
description: Landing page raportu poisson_dixon_coles_baseline — zakres, wnioski, link do HTML i odniesienia do dokumentacji football-data.
keywords: raport, poisson, dixon-coles, baseline, html, marimo, football-data, PIT, grid search
---

# +lucide:file-chart-column+ Raport baseline Poisson Dixon–Coles

Pełna treść raportu (wykresy, tabele, tekst metodyczny) jest w pliku HTML —
poniżej znajdziesz **skrót**, jak raport interpretować i po kolei przez niego
przechodzić.

Wyeksportowany HTML powstaje z notatnika
`notebooks/reports/poisson_dixon_coles_baseline.py` (domyślnie do
`outputs/reports/`; pod build dokumentacji kopiowany jest do `docs/reports/`).

!!! warning "Status raportu"
    To jest raport badawczy dla baseline'u. Wnioski służą do wyboru kolejnych
    eksperymentów, a nie jako gotowa rekomendacja produkcyjna.

!!! tip "Pełny raport"
    Otwórz statyczny raport w przeglądarce:
    [poisson_dixon_coles_baseline.html](poisson_dixon_coles_baseline.html).

## Co zawiera raport

Raport analizuje baseline oparty o `PoissonDixonColesModel`, gdzie lambdy są
odtwarzane z kursów bukmacherskich dla rynków `1X2` oraz Over/Under 2.5. Celem
nie jest jeszcze model produkcyjny, tylko sprawdzenie, jak daleko można dojść
prostym modelem probabilistycznym opartym wyłącznie na kursach.

W raporcie znajdziesz:

- opis transformacji kursów na prawdopodobieństwa i lambdy,
- grid search po `rho` oraz `bias_correction`,
- porównanie metryk `avg_points`, `NLL` i `weighted NLL`,
- diagnostykę PIT i worm ploty dla kalibracji,
- test na aktualnej próbie konkursowej,
- porównanie profilu typowanych wyników z wynikami rzeczywistymi.

## Najważniejsze wnioski

Najlepszy wariant konkursowy w tym raporcie to okolice `rho = -0.18` oraz
`bias_correction = 1.02`, czyli optimum wybrane pod `avg_points` na próbie
historycznej. Na aktualnej próbie konkursowej wariant ten osiąga około **148**
punktów na **163** mecze, ale powierzchnia punktacji jest niestabilna, więc
pojedynczego optimum nie należy traktować jako trwałej prawdy modelowej.

Standardowe `NLL` daje znacznie stabilniejszą powierzchnię optymalizacji, ale w
tym baseline prowadzi do przeciętniejszych wyników konkursowych. To raczej
wskazuje na ograniczenia modelu z jednym globalnym `bias_correction`, a nie na
problem z samą metryką.

`weighted NLL` wypada w raporcie jako antywzorzec: znajduje ekstremalne
parametry (`rho` dodatnie i bardzo niskie lambdy), które matematycznie poprawiają
metrykę, ale psują kalibrację i prowadzą do nadmiernego typowania wyników `1:0`
/ `0:1`.

## Jak czytać raport

Najpierw warto przejść przez sekcję metodologiczną i grid search, a dopiero potem
czytać PIT oraz próbę out-of-sample. Same heatmapy punktów są pomocne, ale bywają
szumne; pełniejszy obraz daje zestawienie ich z `NLL`, PIT i profilem dokładnych
wyników.

Najbardziej praktyczne sekcje:

1. **Grid search** — gdzie leżą optima różnych metryk,
2. **PIT** — czy rozkład bramek jest skalibrowany,
3. **Out-of-sample** — jak warianty zachowują się na aktualnej próbie konkursowej,
4. **Podsumowanie** — decyzje projektowe i dalsze kroki.

## Status rekomendacji

Ten raport traktuj jako baseline i diagnozę problemu, nie jako zamkniętą
procedurę typowania. Najbardziej obiecujący kierunek dalszych prac to odejście
od jednego globalnego mnożnika `bias_correction` i osobne modelowanie
\(\lambda_{\text{home}}\) oraz \(\lambda_{\text{away}}\), na przykład przez
bardziej elastyczny model GLM.

## Kontekst w dokumentacji

- [Trening Poisson Dixon–Coles](../guides/03-training-poisson-dc.md) —
  jak ustawić `PoissonDixonColesModel` i ewaluować predykcje.
- [Korekta Dixona-Colesa](../concepts/dixon-coles-correction.md) —
  parametr `rho` i kalibracja.
- [Grid search i tuning](../guides/06-grid-search-and-tuning.md) —
  siatka po `rho` i `bias_correction`, cache wyników, predictive vs trainable
  grid search.
- [Interpretacja PIT i worm plot](../guides/07-pit-diagnostics-interpretation.md) —
  jak czytać histogramy i worm ploty kalibracji używane w raporcie baseline.
