---
title: Kursy → % implikowane
summary: Dlaczego power method i jak czytać implikowane prawdopodobieństwa
sidebar_title: Kursy → % implikowane
order: 2
description: Intuicja metody potęgowej dla implied probabilities z kursów bukmacherskich w football-data.
keywords: implied probability, power method, overround, odds, football-data
---

# +lucide:percent+ Z kursów do prawdopodobieństw

Kursy dziesiętne (decimal odds) z warstwy danych opisują stawkę zwrotu przy
trafionym typie. Z nich liczymy **implikowane** prawdopodobieństwa wyników, a
następnie — metodą potęgową — usuwamy **marżę bukmachera** (overround), żeby
dostać rozkład sumujący się do 1. Funkcje publiczne:
[`add_power_implied_probabilities`][src.features.add_power_implied_probabilities] i
[`add_power_implied_probabilities_standard_markets`][src.features.add_power_implied_probabilities_standard_markets].

## 1. Od kursu do „surowego” prawdopodobieństwa

Dla rynku z \(n\) wzajemnie wykluczającymi się wynikami (np. \(n=3\) w 1X2)
niech \(o_i > 1\) będzie kursem dziesiętnym na wynik \(i\). Bukmacher
implikuje, że „uczciwa” stawka \(1\) zwróciłaby \(o_i\) przy trafieniu \(i\).
**Surowe** (nieznormalizowane) prawdopodobieństwo definiujemy jako odwrotność
kursu:

\[
q_i = \frac{1}{o_i}, \qquad i = 1,\ldots,n.
\]

Przykład 1X2: kolumny `trimmed_avg_1`, `trimmed_avg_X`, `trimmed_avg_2` →
\(o_1, o_X, o_2\) → \(q_1, q_X, q_2\).

!!! abstract "Co to znaczy w praktyce"
    Kurs \(2.00\) odpowiada \(q = 0{,}50\) (50% „na papierze”).
    Kurs \(1.50\) → \(q \approx 0{,}67\). Im niższy kurs, tym wyższe \(q_i\).

## 2. Overround — dlaczego \(q_i\) nie sumują się do 1

Przy kursach z marżą suma surowych implikowanych prawdopodobieństw jest
**większa** niż 1:

\[
V = \sum_{i=1}^{n} q_i = \sum_{i=1}^{n} \frac{1}{o_i} > 1.
\]

Wielkość \(V - 1\) to nadwyżka (często mówi się o **overround** lub „vigorish”);
czasem podaje się ją jako procent \(\frac{V-1}{1}\times 100\%\). Intuicja:
bukmacher wycenia każdy wynik nieco „taniej”, niż gdyby rynek był idealnie
sprawiedliwy, więc gdybyś obstawiał wszystkie wyniki proporcjonalnie do \(q_i\),
straciłbyś średnio na marży.

Rozkład **prawdopodobieństw** musi spełniać:

\[
\sum_{i=1}^{n} p_i = 1, \qquad p_i \ge 0.
\]

Wektor \(q = (q_1,\ldots,q_n)\) sam w sobie tym warunkiem nie spełnia — stąd
krok „de-vig” w pipeline cech.

## 3. Metoda potęgowa (używana w projekcie)

Zamiast prostego dzielenia przez \(V\) (normalizacja proporcjonalna
\(p_i = q_i / V\)) stosujemy **metodę potęgową**: szukamy wykładnika \(k > 0\),
taki że

\[
\sum_{i=1}^{n} q_i^{\,k} = 1,
\]

a prawdopodobieństwa po usunięciu marży to

\[
p_i = q_i^{\,k} = \left(\frac{1}{o_i}\right)^{k}.
\]

W kodzie (`_power_implied_probabilities_from_odds`) dla każdego wiersza (meczu /
rynek):

1. `raw_probs = 1.0 / odds` → wektor \(q\),
2. jeśli \(\left|\sum_i q_i - 1\right| < \texttt{tolerance}\), zwracane jest
   \(q\) (brak marży w granicy numerycznej),
3. w przeciwnym razie rozwiązywane jest równanie
   \(f(k) = \sum_i q_i^{k} - 1 = 0\) metodą **Newtona** (`scipy.optimize.newton`),
4. wynik: `final_probs = raw_probs ** k_opt`.

!!! info "Równanie w implementacji"
    Docstring [`add_power_implied_probabilities`][src.features.add_power_implied_probabilities]
    podaje je wprost:
    `sum((1/odds)^k) = 1` dla każdego wiersza.

### Intuicja wykładnika \(k\)

- Gdy \(V > 1\) (typowy przypadek), potrzebujesz \(k > 1\): dla \(0 < q_i < 1\)
  podniesienie do \(k>1\) **zmniejsza** każde \(q_i^{\,k}\), więc suma spada z
  \(V\) do 1.
- Gdy \(V = 1\), wystarczy \(k = 1\) i \(p_i = q_i\).
- Dla \(k > 0\) zachowana jest **kolejność** wyników: jeśli \(q_i > q_j\), to
  \(p_i > p_j\) — „faworyt” z kursów zostaje faworytem po de-vig.

Metoda potęgowa nie jest tożsama z \(p_i = q_i/V\); przy tych samych kursach
obie dają nieco inne \(p_i\), ale obie sumują się do 1. W tym repozytorium
świadomie wybrano potęgową (spójność z pipeline lambd Poissona i notatnikami).

### Przykład liczbowy (rynek 2-way)

Kursy BTTS: \(o_{\mathrm{yes}} = 1{,}90\), \(o_{\mathrm{no}} = 1{,}95\).

\[
q_{\mathrm{yes}} = \frac{1}{1{,}90} \approx 0{,}526,\quad
q_{\mathrm{no}} = \frac{1}{1{,}95} \approx 0{,}513,\quad
V \approx 1{,}039.
\]

Solver znajduje \(k \approx 1{,}12\) i np. \(p_{\mathrm{yes}} \approx 0{,}512\),
\(p_{\mathrm{no}} \approx 0{,}488\) — suma 1, relacja „yes bardziej
prawdopodobne niż no” zachowana.

## 4. Rynki wielowynikowe

Ten sam wzór działa dla dowolnego \(n \ge 2\):

| Rynek | Kolumny suffix (przykład) | \(n\) |
| --- | --- | --- |
| 1X2 | `1`, `X`, `2` | 3 |
| BTTS | `btts_yes`, `btts_no` | 2 |
| Over/under 2.5 | `over_25`, `under_25` | 2 |

[`add_power_implied_probabilities`][src.features.add_power_implied_probabilities]
przyjmuje listy `odds_columns` i `output_columns` (domyślnie
`prob_<nazwa_kolumny_kursu>`). Wrapper
[`add_power_implied_probabilities_standard_markets`][src.features.add_power_implied_probabilities_standard_markets]
składa to dla trzech rynków
powyżej z prefiksem kursów (`trimmed_avg`, `avg`, `max`) — patrz
[Budowa cech](../guides/02-building-features.md).

```python
from src.features import add_power_implied_probabilities

df = add_power_implied_probabilities(
    df,
    odds_columns=["trimmed_avg_1", "trimmed_avg_X", "trimmed_avg_2"],
    output_columns=["prob_trimmed_avg_1", "prob_trimmed_avg_X", "prob_trimmed_avg_2"],
    errors="coerce",
)
```

## 5. Parametry i bezpieczeństwo danych

| Parametr | Domyślnie | Znaczenie |
| --- | --- | --- |
| `min_odds` | `1.0` | Każde \(o_i\) musi być **większe** niż ta wartość (inaczej błąd wiersza). |
| `tolerance` | `1e-8` | Jeśli \(\sum q_i \approx 1\), pomijany jest solver — zwracane \(q\). |
| `initial_k`, `max_iter` | `1.0`, `50` | Start i limit iteracji Newtona dla \(k\). |
| `errors` | `"coerce"` | `"coerce"`: zły wiersz → `NaN` w kolumnach prob; `"raise"`: wyjątek z indeksem wiersza. |

!!! warning "Uwaga"
    Kursy \(\le 1\) lub nieskończone / brakujące w wierszu nie przechodzą przez
    solver — używaj `errors="coerce"` w eksploracji, `"raise"` gdy chcesz
    natychmiast złapać brudne dane.

## 6. Co dalej w pipeline

Kolumny `prob_*` są wejściem do lambd Poissona (remis z 1X2 nie wchodzi do
podziału home/away, ale over 2.5 tak) — szczegóły w
[Lambdy Poissona](poisson-lambdas.md).

## Zobacz też

- [API src.features — funkcje power](../api/features.md)
- [Lambdy Poissona](poisson-lambdas.md) (kolejny krok po prawdopodobieństwach)
