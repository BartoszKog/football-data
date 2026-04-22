---
title: Lambdy Poissona
summary: Jak z P(1), P(2), P(over 2.5) liczymy baseline_lambda i calibrated_lambda
sidebar_title: Lambdy Poissona
order: 3
description: Baseline i skalibrowane lambdy Poissona — wzory, domyślne kolumny i parametry, zgodnie z poisson_priors.py i notatnikiem baseline DC.
keywords: poisson lambda, baseline, calibration, over 2.5, poisson_priors, football-data
---

# +lucide:sigma+ Lambdy Poissona z rynków 1X2 i over 2.5

W modelu scoreline (np. Poisson + Dixon–Coles) liczby goli gospodarza i gościa
traktujesz jako **niezależne** Poissony z parametrami \(\lambda_{home}\),
\(\lambda_{away}\). W tym projekcie te \(\lambda\) **nie** pochodzą z historii
meczy, tylko z **prawdopodobieństw implikowanych z kursów** (1X2 + over 2.5).

Intuicja krok po kroku (łącznie z macierzą Poissona i korektą DC) jest
rozpisana w notatniku Marimo
`notebooks/reports/poisson_dixon_coles_baseline.py` — sekcja *„Jak działa
algorytm”*; poniżej skupiamy się na **kroku 1**: odtworzeniu lambd w kodzie
[`add_baseline_poisson_lambdas`][src.features.poisson_priors.add_baseline_poisson_lambdas]
i [`add_calibrated_poisson_lambdas`][src.features.poisson_priors.add_calibrated_poisson_lambdas].

## 1. Baseline — skąd bierze się \(\lambda_{home}\) i \(\lambda_{away}\)

Implementacja w repozytorium: `src/features/poisson_priors.py`.

Dla każdego wiersza (meczu) masz trzy liczby w \([0, 1]\):

- \(p_{home}\), \(p_{away}\) — prawdopodobieństwa wygranej gospodarza / gościa
  (zwykle z power-implied; **remis nie wchodzi** do podziału całkowitego \(\lambda\)),
- \(P_{>2.5}\) — prawdopodobieństwo ponad 2.5 gola w meczu.

### Całkowita oczekiwana liczba bramek \(\lambda_{total}\)

Zakładamy, że **suma** goli w meczu ma rozkład Poissona z parametrem
\(\lambda_{total}\). Wtedy

\[
P_{>2.5} = P(\text{suma} > 2) = 1 - \sum_{k=0}^{2} \frac{\lambda_{total}^k e^{-\lambda_{total}}}{k!}.
\]

W kodzie budowana jest **siatka** wartości \(\lambda\) na odcinku
`[lambda_min, lambda_max]` (domyślnie od `0.01` do `20.0`, `grid_size=10000`),
dla każdej wartości liczone jest \(P_{>2.5}\) z CDF Poissona, a następnie
**interpolacja odwrotna**: z obserwowanego \(P_{>2.5}\) z kolumny odczytywane
jest \(\lambda_{total}\) (wartości brzegowe są lekko przycinane, żeby uniknąć
ekstrapolacji poza siatkę).

Na koniec stosowany jest mnożnik **`bias_correction`** (domyślnie `1.035` w
baseline, `1.0` w ścieżce skalibrowanej — patrz niżej):

\[
\lambda_{total} \leftarrow \lambda_{total} \cdot \text{bias\_correction}.
\]

### Podział na gospodarza i gościa

Remis nie jest używany do udziałów — tylko stosunek „wygrana home” vs „wygrana away”:

\[
\lambda_{home} = \lambda_{total} \cdot \frac{p_{home}}{p_{home} + p_{away}}, \qquad
\lambda_{away} = \lambda_{total} \cdot \frac{p_{away}}{p_{home} + p_{away}}.
\]

Jeśli \(p_{home} + p_{away} \le 0\) lub dane są niespójne, przy `errors="coerce"`
(wartość domyślna) dostaniesz `NaN` w odpowiednich wierszach.

### Domyślne kolumny wejścia / wyjścia (baseline)

| Rola | Domyślna nazwa kolumny |
| --- | --- |
| Wejście \(p_{home}\) | `prob_trimmed_avg_1` |
| Wejście \(p_{away}\) | `prob_trimmed_avg_2` |
| Wejście \(P_{>2.5}\) | `prob_trimmed_avg_over_25` |
| Wyjście | `baseline_lambda_home`, `baseline_lambda_away` |

To są **atrybuty** (nowe kolumny) dodawane do kopii `DataFrame`; typ to zwykle
`float64`, wartości rzędu \(0.5\)–\(3\) dla typowych meczów ligowych.

## 2. Calibrated — korekta względem baseline

[`add_calibrated_poisson_lambdas`][src.features.poisson_priors.add_calibrated_poisson_lambdas]
najpierw liczy baseline **do kolumn tymczasowych**, potem stosuje na każdą stronę
**to samo** przekształcenie wykładnicze:

\[
\lambda^{cal}_{home} = \exp(B_0 + B_1 \cdot \lambda^{base}_{home}), \qquad
\lambda^{cal}_{away} = \exp(B_0 + B_1 \cdot \lambda^{base}_{away}),
\]

gdzie domyślnie \(B_0 = -0.354611\), \(B_1 = 0.443665\) (dopasowanie z
`02_GAM_Lab`, opis w docstringu modułu). Tu \(B_1\) mnoży **samą**
baseline lambdę, nie \(\log \lambda\).

| Rola | Domyślna nazwa |
| --- | --- |
| Wyjście | `calibrated_lambda_home`, `calibrated_lambda_away` |
| `bias_correction` (z baseline) | domyślnie `1.0` (kalibracja ma zastąpić stary „ręczny” mnożnik) |

## 3. Przykład w kodzie

Najpierw prawdopodobieństwa (jak w [Budowa cech](../guides/02-building-features.md)),
potem baseline i opcjonalnie calibrated:

```python
from src.features import (
    add_power_implied_probabilities_standard_markets,
    add_baseline_poisson_lambdas,
    add_calibrated_poisson_lambdas,
)

df = add_power_implied_probabilities_standard_markets(
    df,
    odds_prefix="trimmed_avg",
    output_prefix="prob_trimmed_avg",
)

df = add_baseline_poisson_lambdas(
    df,
    prob_home_col="prob_trimmed_avg_1",
    prob_away_col="prob_trimmed_avg_2",
    prob_over25_col="prob_trimmed_avg_over_25",
    # bias_correction=1.035  # domyślnie w add_baseline_poisson_lambdas
)

df = add_calibrated_poisson_lambdas(df)
# Domyślnie: intercept=-0.354611, slope=0.443665, bias_correction=1.0
```

Fragment typowego wiersza po obu krokach (nazwy kolumn — **atrybuty** modelu
cech):

```text
baseline_lambda_home     1.37
baseline_lambda_away     1.05
calibrated_lambda_home   1.12
calibrated_lambda_away   0.94
```

(dokładne liczby zależą od kursów meczu; to tylko ilustracja rzędu wielkości).

Szybki podgląd w notatniku:

```python
cols = [
    "baseline_lambda_home",
    "baseline_lambda_away",
    "calibrated_lambda_home",
    "calibrated_lambda_away",
]
df[cols].describe()
```

Kolejny krok w pipeline scoreline (macierz z \(\lambda\) i korektą DC, potem
wybór typu pod oczekiwane punkty): [Oczekiwane punkty i wybór wyniku](expected-points-optimization.md).

## Zobacz też

- [Oczekiwane punkty i wybór wyniku](expected-points-optimization.md)
- [API — `add_baseline_poisson_lambdas` / `add_calibrated_poisson_lambdas`](../api/features.md)
- [Kursy → prawdopodobieństwa](odds-to-probabilities.md)
- Notatnik baseline DC: `notebooks/reports/poisson_dixon_coles_baseline.py` (uruchom przez Marimo w katalogu repo)
