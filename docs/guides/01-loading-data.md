---
title: Ładowanie danych
summary: Od plików JSON w data/raw do DataFrame z kolumnami kursów — src.data
sidebar_title: Ładowanie danych
order: 1
description: Przewodnik po wczytywaniu surowych sezonów, schemacie kolumn i budowie kolumn kursowych w football-data.
keywords: src.data, load_raw_seasons, odds, trimmed_avg, schema, columns, football-data
---

# +lucide:database+ Ładowanie i przygotowanie danych

Warstwa [`src.data`](../api/data.md) zamienia pliki z OddsHarvestera na jeden
DataFrame gotowy do feature engineeringu i modeli.

## Kiedy która funkcja

- `load_raw_seasons` — Gdy masz już `DataFrame` lub chcesz tylko surowe wiersze przed dodaniem kursów.
- `load_and_add_odds_columns` — Pełny pipeline: glob → surowe wiersze → kolumny kursów (pozostają kolumny rynków).
- `load_and_add_odds_columns_compact` — Jak wyżej, ale **usuwa** surowe kolumny rynków po wyliczeniu kursów — wygodniejsze pod modele.

Szczegóły sygnatur: [API src.data](../api/data.md).

## +lucide:table-properties+ Schemat: co jest w `df`

Pipeline ma dwa wyraźne etapy — warto je rozróżnić w notatniku.

### Po `load_raw_seasons`

- Kolumny odpowiadają **kluczom obiektów JSON** z scrapera (jeden wiersz = jeden mecz). Dokładny zestaw zależy od wersji OddsHarvestera — zawsze możesz sprawdzić `df.columns` po wczytaniu własnych plików.
- Loader **dokleja** metadane:
  - `season` — etykieta sezonu z nazwy pliku (patrz niżej),
  - `source_file` — nazwa pliku źródłowego.
- Normalizacja typów (gdy kolumny istnieją):
  - `match_date` — `datetime`, z UTC na **Europe/Warsaw**,
  - `scraped_date` — `datetime` w **UTC**,
  - `home_score`, `away_score` — wartości numeryczne (`NaN` przy błędach konwersji).
- Jeśli jest `match_link`, wiersze są **deduplikowane** po tym polu: zostaje wpis z najnowszym `scraped_date` (przy remisie — ostatni w sortowaniu).

W praktyce w projekcie pojawiają się m.in. identyfikatory i wynik (`match_link`, `match_date`, `home_team`, `away_team`, `home_score`, `away_score`) oraz trzy kolumny z surowymi listami kursów bukmacherskich — patrz następna podsekcja.

### Surowe kolumny rynków (przed agregacją)

W komórce DataFrame pole rynku to zwykle **lista słowników** (jeden element = jeden bukmacher) albo string z listą, którą kod parsuje przy agregacji. Nazwy pól wewnątrz słownika muszą być zgodne z konfiguracją w kodzie (`1`, `X`, `2`, `btts_yes`, `btts_no`, `odds_over`, `odds_under`).

| Kolumna w `df` | Znaczenie |
| --- | --- |
| `1x2_market` | Kursy 1 / X / 2 |
| `btts_market` | BTTS yes / no |
| `over_under_2_5_market` | Over / under 2.5 (`odds_over`, `odds_under` w JSON) |

### Po dodaniu kursów (`add_odds_columns` / `load_and_add_odds_columns*`)

Dla każdej selekcji w rynku powstają trzy kolumny: `max_*`, `avg_*`, `trimmed_avg_*`. Parametr `trim_drop` steruje obcięciem skrajnych wartości przy `trimmed_avg_*`.

| Rynek (kolumna źródłowa) | Sufiksy w `max_*` / `avg_*` / `trimmed_avg_*` |
| --- | --- |
| `1x2_market` | `1`, `X`, `2` |
| `btts_market` | `btts_yes`, `btts_no` |
| `over_under_2_5_market` | `over_25`, `under_25` |

Pełny zestaw nazw wygenerowanych przez warstwę danych (przy `odds_metrics=None`) to m.in. `max_1`, `avg_1`, `trimmed_avg_1`, … dla wszystkich selekcji z tabeli.

Wariant **`_compact`** usuwa kolumny `1x2_market`, `btts_market`, `over_under_2_5_market`.  
`load_and_add_odds_columns_compact(..., odds_metrics="trimmed_avg")` zostawia tylko kolumny z prefiksem `trimmed_avg_`.

!!! tip "Szybka diagnostyka w notatniku"
    Po wczytaniu: `df.head(3).T` albo `sorted(df.columns)` — zobaczysz faktyczny zestaw pól z Twoich plików JSON oraz wygenerowane `*_` kursy.

Dalsza obróbka kursów na prawdopodobieństwa: [Od kursów do prawdopodobieństw](../concepts/odds-to-probabilities.md).

## +lucide:file-json+ Pliki sezonów i etykieta `season`

Domyślny glob to `data/raw/1liga_*.json`. Z nazwy pliku:

- `1liga_20192020.json` → `season == "2019/2020"`,
- plik ze stemem kończącym się na `_current` → `season == "current"`,
- inne nazwy → `season` ustawiane jest na `stem` pliku (bez rozszerzenia).

## Przykład: compact + jeden typ kursów

Typowy start w notatniku — wzór jak w [Getting started](../getting-started.md):

```python
from src.data import load_and_add_odds_columns_compact

df = load_and_add_odds_columns_compact(
    pattern="data/raw/1liga_*.json",
    trim_drop=1,
    odds_metrics="trimmed_avg",
)
```

`trim_drop` kontroluje obcięcie skrajnych kursów przy `trimmed_avg_*`.
`odds_metrics` może ograniczyć wygenerowane kolumny (np. tylko `trimmed_avg`).

## Sezony i holdout

Przed modelem historycznym **odfiltruj** sezon roboczy / `current`, żeby nie
mieszać niedokończonej rundy z treningiem:

```python
df_hist = df[df["season"] != "current"].copy()
```

## Skąd wziąć pliki

- Instalacja scrapera i batch: [Getting started — sekcja o danych](../getting-started.md#3-pobierz-surowe-dane).
- Ponawianie pojedynczych meczów: `scripts/retry_failed_matches.py`,
  `scripts/manual_retry_matches.py`.

## Zobacz też

- [Od kursów do prawdopodobieństw](../concepts/odds-to-probabilities.md)
- [Budowa cech](02-building-features.md)
- [Struktura projektu](../project-structure.md)
