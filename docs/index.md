---
title: Football Data Docs
summary: Osobista dokumentacja modeli Poisson / Dixon-Coles i XGBoost do Supertypera
sidebar_title: Home
order: 0
description: Dokumentacja projektu football-data - modele statystyczne i ML do typowania wyników 1. ligi.
keywords: football-data, mkdocs, supertyper, poisson, dixon-coles, xgboost
---

# Football Data Docs

Osobista dokumentacja projektu `football-data` — modeli statystycznych
i ML do typowania wyników piłkarskich w konkursie [Supertyper](https://www.app-helper.com/betting_game/?appid=50832).

Tę dokumentację traktuj jako **szybkie przypomnienie** po kilku tygodniach
przerwy od projektu: skąd wziąć dane, jak uruchomić pipeline, czym różnią się
warianty poszczególnych funkcji i dlaczego pewne decyzje (np. konkretna wartość
`rho` w korekcie Dixona-Colesa) zostały podjęte.

## +lucide:compass+ Jak korzystać

- Jeśli wracasz do projektu po przerwie — zacznij od [Getting started](getting-started.md).
- Jeśli wiesz co chcesz zrobić, ale nie pamiętasz jak — zajrzyj do [Guides](guides/01-loading-data.md).
- Jeśli chcesz przypomnieć sobie "czemu tak, a nie inaczej" — zajrzyj do [Concepts](concepts/dixon-coles-correction.md).
- Jeśli szukasz sygnatury konkretnej funkcji lub klasy — zajrzyj do [API](api/data.md).

## +lucide:map+ Mapa dokumentacji

### +lucide:rocket+ Start

- [Getting started](getting-started.md) — instalacja, OddsHarvester, pobranie
  danych, uruchomienie pierwszego notatnika i dokumentacji offline.
- [Struktura projektu](project-structure.md) — mapa katalogów (`data/`, `src/`,
  `notebooks/`, skrypty) i skrót warstw kodu.

### +lucide:book-open+ Guides (jak coś zrobić)

- [Ładowanie danych](guides/01-loading-data.md) — `src.data`, kolumny kursów,
  `trimmed_avg`, filtrowanie sezonów.
- [Budowa cech](guides/02-building-features.md) — prawdopodobieństwa
  implikowane, lambdy Poissona.
- [Trening Poisson Dixon–Coles](guides/03-training-poisson-dc.md) —
  `PoissonDixonColesModel`, predykcja, ewaluacja.
- [Trening XGBoost Poisson](guides/04-training-xgboost.md) — `fit` / `eval_df`,
  cechy, tuning sezonowy.
- [Ewaluacja predykcji](guides/05-evaluating-predictions.md) — metryki,
  wykresy, deviance.
- [Grid search i tuning](guides/06-grid-search-and-tuning.md) — 1D/2D grid
  search, cache, trainable grid search z walk-forward po sezonach.
- [Interpretacja PIT i worm plot](guides/07-pit-diagnostics-interpretation.md) —
  jak czytać kształty diagnostyczne dla dobrej kalibracji, biasu lambd i
  over/underconfidence.

### +lucide:lightbulb+ Concepts (dlaczego tak)

- [Korekta Dixona-Colesa](concepts/dixon-coles-correction.md) — intuicja za
  parametrem `rho`, kalibracja NLL vs grid search po `avg_points`.
- [Kursy → prawdopodobieństwa](concepts/odds-to-probabilities.md) — power
  method, rynki 1X2 i 2-way.
- [Lambdy Poissona](concepts/poisson-lambdas.md) — baseline i kalibracja.
- [Oczekiwane punkty i wybór wyniku](concepts/expected-points-optimization.md) —
  od macierzy \(P_{DC}\) do `pred_xpts` i typu \(h{:}a\).
- [Walidacja sezonowa](concepts/season-walk-forward-validation.md) — train /
  val / eval przy tuningu XGBoost.
- [Punktacja i metryki](concepts/scoring-rules.md) — Supertyper, `avg_points`
  vs `total_points`.

### +lucide:code+ API (auto-generowane z docstringów)

- [`src.data`](api/data.md) — wczytywanie surowych sezonów i budowa kolumn kursowych.
- [`src.features`](api/features.md) — prawdopodobieństwa implikowane, lambdy Poissona.
- [`src.models`](api/models.md) — modele predykcyjne (`PoissonDixonColesModel`,
  `XGBoostPoissonModel`) i kontrakty interfejsów.
- [`src.models.tuning`](api/tuning.md) — grid search (predictive i sezonowy
  trainable), `make_season_walk_forward_splits`.
- [`src.models.evaluation`](api/evaluation.md) — punktacja Supertyper, Poisson
  deviance, wizualizacje ewaluacji.

## +lucide:languages+ Konwencja językowa

- Guides, Concepts, Getting started — **po polsku** (narracja projektu).
- API — **po angielsku** (docstringi z kodu, spójne z `src/`).
- Kolumny DataFrame, nazwy parametrów, logi — **angielskie**, nie tłumaczone.
