---
title: Oczekiwane punkty i wybór wyniku
summary: Macierz vs kandydaci h:a, ogony rozkładu, ExpectedPointsOptimizer, pred_xpts
sidebar_title: Oczekiwane punkty
order: 4
description: Macierz scoreline, max_goals_matrix vs max_goals_prediction, oczekiwane punkty, pred_xpts w football-data.
keywords: ExpectedPointsOptimizer, max_goals_matrix, max_goals_prediction, PoissonMatrixBuilder, pred_xpts, football-data
---

# +lucide:target+ Oczekiwane punkty i wybór wyniku

Modele scoreline w tym projekcie (`PoissonDixonColesModel`, `XGBoostPoissonModel`)
nie zwracają „najbardziej prawdopodobnego” wyniku \(i{:}j\) z macierzy. Zamiast
tego wybierają typ \((h,a)\), który **maksymalizuje oczekiwaną liczbę punktów**
Supertypera przy niepewnym faktycznym wyniku — to jest kolumna `pred_xpts`.

Poniżej: skąd bierze się macierz, jak liczy się \(\mathbb{E}[\text{punkty}]\) i
jak spiąć to z [punktacją 3/2/1/0](scoring-rules.md).

## Rola macierzy prawdopodobieństw

Dla jednego meczu [`PoissonMatrixBuilder`](../api/models.md) buduje kwadratową
macierz \(P(i,j)\): prawdopodobieństwo, że **faktyczny** wynik to \(i\) goli
gospodarza i \(j\) goli gościa, na siatce \(i,j \in \{0,\ldots,G\}\) z
\(G =\) `max_goals_matrix`.

1. Bazowo: niezależne Poissony z \(\lambda_{home}\), \(\lambda_{away}\) (skąd
   brać \(\lambda\) — [Lambdy Poissona](poisson-lambdas.md)).
2. Na komórkach \((0,0)\), \((0,1)\), \((1,0)\), \((1,1)\) nakładana jest korekta
   Dixona-Colesa \(\tau(i,j;\rho,\lambda)\) — [Korekta Dixona-Colesa](dixon-coles-correction.md).
3. Wartości są **normalizowane** do rozkładu prawdopodobieństwa (w kodzie:
   `clip`, suma, dzielenie przez sumę — `src/models/components/matrix_builders.py`).

Macierz opisuje więc **niepewność rzeczywistego wyniku**, a nie Twój typ.

## `max_goals_matrix` vs `max_goals_prediction`

- **`max_goals_matrix`** (`G`) — rozmiar macierzy \(P\): indeksy faktycznego wyniku
  \((i,j)\) to \(0,\ldots,G\). Im większe \(G\), tym więcej masy rozkładu
  mieścisz w modelu (kosztem rozmiaru tablicy).
- **`max_goals_prediction`** — górna granica **kandydatów na typ** \((h,a)\) w
  optymalizatorze; musi być \(\le G\) (tak jest w konstruktorach modeli).
  Typowy sens: przeszukujesz tylko „rozsądne” wyniki (np. do 4 goli na drużynę),
  podczas gdy macierz liczy prawdopodobieństwa na nieco szerszej siatce (np. do 6).

Przy **normalizacji** \(P\) suma prawdopodobieństw w komórkach \(0..G\) jest 1,
ale część masy **prawdziwego** rozkładu Poissona leży poza tą siatką (ogon).
Przy typowych \(\lambda\) w lidze i domyślnym \(G=6\) jest to zwykle pomijalne;
bardzo małe \(G\) mogą zniekształcić zarówno \(P\), jak i \(\mathbb{E}[\text{punkty}]\).

## Oczekiwane punkty dla jednego kandydata \((h,a)\)

Twój typ to para nieujemnych całkowitych bramek \((h,a)\) z dozwolonego zakresu
kandydatów `0…max_goals_prediction`. Dla ustalonej macierzy \(P\):

\[
\mathbb{E}[\text{punkty} \mid \text{typ } h{:}a]
  = \sum_{i,j} P(i,j) \cdot \text{punkty}(h, a;\, i, j),
\]

gdzie \(\text{punkty}(h,a;\,i,j)\) to punkty Supertypera za typ \((h,a)\) gdy
faktyczny wynik to \((i,j)\) — ta sama logika co [`score_single_prediction`](../api/evaluation.md)
(tiery: dokładny wynik, różnica bramek, 1X2, pudło). Wagi 3 / 2 / 1 / 0 są
konfigurowalne jako [`ExpectedPointsRule`](../api/models.md); domyślnie
zgodne z [`ScoreRule`](../api/evaluation.md) — patrz
[Punktacja i metryki](scoring-rules.md).

**Intuicja:** nie wybierasz \(\arg\max_{i,j} P(i,j)\), tylko typ, który **średnio**
najlepiej punktuje, gdy rzeczywisty wynik losuje się według \(P\).

## Jak to robi `ExpectedPointsOptimizer`

[`ExpectedPointsOptimizer`](../api/models.md) wstępnie buduje tensor punktów
kształtu `(pred_home, pred_away, real_home, real_away)`, potem jednym
`np.tensordot` z macierzą \(P\) otrzymuje **oczekiwane punkty** dla każdego
kandydata \((h,a)\). Wybór to \(\arg\max\) po siatce kandydatów; zwracane
`pred_xpts` to wartość maksymalna.

Szczegóły implementacji i tie-break — docstring w `src/models/components/optimizers.py`.

## Remisy przy równych oczekiwanych punktach

Gdy kilka par \((h,a)\) ma (z tolerancją numeryczną) tę samą maksymalną
\(\mathbb{E}[\text{punkty}]\), optymalizator:

1. Preferuje kandydata z większym **bezpośrednim** \(P(h,a)\) (prawdopodobieństwo
   dokładnie tego wyniku).
2. Przy dalszym remisie wybór jest deterministyczny po kolejności indeksów
   (najniższe `h`, potem `a`).

Krótki opis także w [Punktacja i metryki](scoring-rules.md#remisy-przy-wyborze-wyniku-z-macierzy).

## Ten sam komponent w obu modelach

`PoissonDixonColesModel` ( \(\lambda\) z rynków) i `XGBoostPoissonModel`
( \(\lambda\) z regresorów) używają tej samej ścieżki: zbuduj macierz →
`ExpectedPointsOptimizer.optimize(matrix)` → `pred_home_goals`, `pred_away_goals`,
`pred_xpts`.

## Zobacz też

- [Trening Poisson Dixon–Coles](../guides/03-training-poisson-dc.md) — kolumny `predict`
- [Trening XGBoost Poisson](../guides/04-training-xgboost.md) — ten sam wybór wyniku po \(\lambda\) z drzew
- [Korekta Dixona-Colesa](dixon-coles-correction.md)
- [Lambdy Poissona](poisson-lambdas.md)
- [Punktacja i metryki](scoring-rules.md)
- [API `src.models`](../api/models.md)
- [API `src.models.evaluation`](../api/evaluation.md) — `score_single_prediction` i tiery punktów
