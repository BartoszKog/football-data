---
title: src.data
summary: Wczytywanie surowych sezonow i budowa kolumn kursowych
sidebar_title: src.data
order: 1
description: API modulu src.data - load_raw_seasons, add_odds_columns, load_and_add_odds_columns_compact.
keywords: src.data, odds, raw seasons, match_link, oddsharvester
---

# +lucide:database+ API — `src.data`

Helpers do wczytywania surowych sezonów JSON z OddsHarvestera, deduplikacji
meczów po `match_link`, normalizacji `match_date` do `Europe/Warsaw` i budowy
kolumn kursowych (`max_*`, `avg_*`, `trimmed_avg_*`).

Zobacz też: [Getting started — dane](../getting-started.md#3-pobierz-surowe-dane).

---

::: src.data.load_raw_seasons

::: src.data.add_odds_columns

::: src.data.add_odds_columns_compact

::: src.data.load_and_add_odds_columns

::: src.data.load_and_add_odds_columns_compact
