---
title: Punktacja i metryki
summary: Reguły 3/2/1/0, ScoreRule vs ExpectedPointsRule, evaluate_score_predictions, tuning
sidebar_title: Punktacja i metryki
order: 6
description: Schemat Supertyper w kodzie — kolejność tierów, 1X2, ScoreRule, metryki evaluate_score_predictions, avg_points vs total_points.
keywords: supertyper, ScoreRule, ExpectedPointsRule, avg_points, total_points, scoring, football-data
---

# +lucide:trophy+ Punktacja Supertyper i metryki ewaluacji

W konkursie typowania stosowany jest schemat **3 / 2 / 1 / 0** punktów za mecz
(w zależności od trafienia dokładnego wyniku, różnicy bramek lub samego 1X2).
W repozytorium odpowiada mu domyślna [`ScoreRule`](../api/evaluation.md) oraz
funkcje [`score_single_prediction`](../api/evaluation.md),
[`compute_points_per_match`](../api/evaluation.md),
[`evaluate_score_predictions`](../api/evaluation.md).

## Kolejność reguł (wyłączne „szuflady”)

`score_single_prediction` stosuje **wzajemnie wykluczające się** poziomy
w **tej** kolejności (pierwsze dopasowanie wygrywa):

1. **Dokładny wynik** — przewidziane `pred_home` / `pred_away` równe faktycznym.
2. **Trafiona różnica bramek** — bez punktu 1: \((pred\_home - pred\_away) = (actual\_home - actual\_away)\).
3. **Trafione 1X2** — bez punktów 1–2: ten sam znak różnicy bramek (wygrana gospodarza / remis / wygrana gościa).
4. **Pudło** — w przeciwnym razie **0** punktów (przy domyślnej regule).

## Jak rozumiane jest 1X2 w kodzie

Nie ma osobnych etykiet „1”, „X”, „2”. Wynik meczu i typ są porównywane przez
**znak różnicy bramek** (`np.sign(pred_home - pred_away)` vs
`np.sign(actual_home - actual_away)`). Remis (w tym 0:0) ma znak **0** po obu
stronach, więc remis jest spójnie „trafionym X”.

## `ScoreRule` — domyślnie 3 / 2 / 1 / 0, da się nadpisać

[`ScoreRule`](../api/evaluation.md) to zamrożony dataclass z polami:

- `exact` (domyślnie 3),
- `goal_diff` (2),
- `outcome` (1),
- `miss` (0).

Możesz podać własną instancję do `evaluate_score_predictions(..., rule=...)` /
`compute_points_per_match(..., rule=...)`, żeby ćwiczyć inną skalę bez zmiany
logiki tierów.

## Dwa typy reguł: `ScoreRule` vs `ExpectedPointsRule`

**Ta sama idea punktów** (te cztery liczby), ale **inne miejsce użycia**:

W projekcie wyróżniamy dwa typy reguł punktacji, z których każdy pełni inną rolę:

- **[`ScoreRule`](../api/evaluation.md):** Służy do ewaluacji po fakcie, czyli liczenia punktów za już wybrany wynik. Używana jest przez takie funkcje jak `score_single_prediction` czy też agregaty w `evaluate_score_predictions`.

- **[`ExpectedPointsRule`](../api/models.md):** W tej regule te same wagi punktacyjne są wykorzystywane przy wyborze prognozowanego wyniku na podstawie macierzy prawdopodobieństw. Trafia ona do [`ExpectedPointsOptimizer`](../api/models.md): predykcja jest tak wybierana, aby maksymalizować **oczekiwane** punkty (`pred_xpts` w `predict`). Opis szczegółowego sposobu obliczania znajdziesz w rozdziale: [Oczekiwane punkty i wybór wyniku](expected-points-optimization.md).

Podsumowując: `ScoreRule` wykorzystywany jest do oceny już postawionych typów, a `ExpectedPointsRule` — do wyboru najlepszego typu z macierzy wyników przewidywanych przez model. W typowym pipeline oba używają tych samych domyślnych wag (3 / 2 / 1 / 0), co zapewnia spójność strategii i oceny.

W typowym pipeline oba używają domyślnych **3 / 2 / 1 / 0**, więc strategia
wyboru wyniku jest spójna z późniejszą oceną.

### Remisy przy wyborze wyniku z macierzy

Gdy kilka kandydatów ma **tę samą** maksymalną wartość oczekiwanych punktów,
`ExpectedPointsOptimizer` rozstrzyga remis: najpierw wyższe prawdopodobieństwo
„trafienia” w dany wynik w macierzy, potem deterministyczna kolejność indeksów
(najniższe `pred_home`, potem `pred_away`). Szczegóły: docstring w `src`.

## Przykłady punktów (domyślna `ScoreRule`)

| Predykcja | Faktyczny | Punkty | Uzasadnienie |
| --- | --- | --- | --- |
| 2 : 1 | 2 : 1 | 3 | Dokładny wynik |
| 3 : 1 | 2 : 0 | 2 | Ta sama różnica (+2), inny wynik |
| 1 : 1 | 2 : 2 | 2 | Remis vs remis — ta sama różnica 0 |
| 2 : 1 | 3 : 1 | 1 | Tylko 1X2 (wygrana gospodarza; różnica +1 vs +2) |
| 2 : 1 | 0 : 1 | 0 | Pudło |

## `evaluate_score_predictions`: słownik metryk

Funkcja zwraca m.in.:

| Klucz | Znaczenie |
| --- | --- |
| `matches_evaluated` | Liczba wierszy po odrzuceniu braków w wymaganych kolumnach |
| `total_points` | Suma punktów |
| `avg_points` | Średnia punktów na mecz |
| `exact_hit_rate` | Ułamek meczów w tierze 1 |
| `goal_diff_hit_rate` | Tier 2 (**bez** tieru 1) |
| `outcome_hit_rate` | Tier 3 (**bez** 1 i 2) |
| `miss_rate` | Reszta (tier 4) |

Cztery ostatnie wskaźniki są **rozłączne** — każdy mecz wpada dokładnie do
jednej kategorii, tak jak w implementacji masek w
[`evaluate_score_predictions`](../api/evaluation.md).

## Dane wejściowe: braki i zaokrąglenie

Wiersze z `NaN` w którejkolwiek z kolumn predykcji lub faktu są **pomijane**.
Wartości liczbowe są wymuszane przez `to_numeric`, a przed punktacją prognozy
i fakty są **zaokrąglane do najbliższej liczby całkowitej** (`rint`), żeby
uniknąć artefaktów typu `2.1 : 0`.

## Po co `avg_points` i `total_points`

- **`avg_points`** — średnia punktów na mecz; **nie zależy** od liczby meczów w
  próbce. Dobry domyślny cel przy tuningu na wielu sezonach.
- **`total_points`** — suma punktów; na krótkim oknie (np. jeden sezon)
  odpowiada bezpośrednio „ile punktów zebrałbyś w tabeli”, ale porównania między
  oknami o różnej liczbie meczów mogą być mylące.

W grid searchu wybierasz `score_key` lub własne `metric_fn` — patrz
[Grid search — metryka rankingowa](../guides/06-grid-search-and-tuning.md#4-metryka-rankingowa-avg_points-vs-total_points-vs-wasna).

## Zobacz też

- [Oczekiwane punkty i wybór wyniku](expected-points-optimization.md)
- [Ewaluacja predykcji](../guides/05-evaluating-predictions.md)
- [API `src.models.evaluation`](../api/evaluation.md)
- [API `src.models` (optymalizator)](../api/models.md)
- [Korekta Dixona-Colesa](dixon-coles-correction.md) — `rho` pod `avg_points` vs NLL
