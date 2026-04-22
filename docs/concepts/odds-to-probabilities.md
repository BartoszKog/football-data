---
title: Kursy → % implikowane
summary: Dlaczego power method i jak czytać implikowane prawdopodobieństwa
sidebar_title: Kursy → % implikowane
order: 2
description: Intuicja metody potęgowej dla implied probabilities z kursów bukmacherskich w football-data.
keywords: implied probability, power method, overround, odds, football-data
---

# +lucide:percent+ Z kursów do prawdopodobieństw

Kursy zawierają **marżę bukmachera** (overround): odwrotności kursów sumują się
do więcej niż 100%, więc nie są od razu spójnym rozkładem prawdopodobieństwa.

## Co robi `add_power_implied_probabilities`

Metoda **potęgowa** szuka wykładnika \(k\), takiego że po podniesieniu każdego
„surowego” prawdopodobieństwa \(1/\text{odds}\) do potęgi \(k\) i
znormalizowaniu dostajesz rozkład, który:

- sumuje się do 1,
- zachowuje sensowny kształt względem hierarchii kursów.

Parametry jak `min_odds` i `errors="coerce"` chronią przed patologicznymi
wartościami w danych.

## Rynki 1X2 i 2-way

Ten sam schemat działa dla trzech wyników (`1`, `X`, `2`) i dla rynków
dwustanowych (np. BTTS yes/no) — podajesz listy kolumn wejściowych i wyjściowych.
Wygodny wrapper: `add_power_implied_probabilities_standard_markets` w
[Budowa cech](../guides/02-building-features.md).

## Zobacz też

- [API src.features — funkcje power](../api/features.md)
- [Lambdy Poissona](poisson-lambdas.md) (kolejny krok po prawdopodobieństwach)
