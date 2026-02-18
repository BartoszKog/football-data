# Football Data

## 📂 Organizacja Projektu

    football-data/
    ├── README.md             <- Główny opis projektu, jak uruchomić, skąd wziąć dane.
    ├── data/
    │   ├── external/         <- Dane zewnętrzne (np. słowniki).
    │   ├── raw/              <- Surowe, niezmieniane dane (JSON-y z OddsHarvester).
    │   └── processed/        <- Czyste dane gotowe do modelowania (wynik działania skryptów z src).
    │
    ├── notebooks/            <- Notatniki Marimo/Jupyter.
    │   ├── exploration/      <- Eksperymenty, sprawdzanie danych, brudnopisy badawcze.
    │   └── reports/          <- Gotowe, "wyczyszczone" analizy prezentujące wyniki (np. dla GitHub Pages).
    │
    ├── outputs/              <- Wygenerowane pliki.
    │   ├── figures/          <- Wykresy i grafiki używane w raportach (png, svg).
    │   └── reports/          <- Wyeksportowane raporty (np. HTML, PDF).
    │
    ├── sandbox/              <- Kod tymczasowy, testy, skrypty "na raz" (ignorowane przez git).
    │
    ├── scripts/              <- Skrypty uruchomieniowe/automatyzacyjne (np. update_data.py).
    │
    ├── src/                  <- Kod źródłowy projektu (moduły Python).
    │   ├── data/             <- Skrypty do pobierania i przetwarzania danych.
    │   ├── features/         <- Skrypty do inżynierii cech (np. wyliczanie formy drużyny).
    │   ├── models/           <- Skrypty do trenowania modeli i predykcji.
    │   └── visualization/    <- Funkcje pomocnicze do tworzenia wykresów.
    │
    └── OddsHarvester/        <- Zewnętrzne narzędzie do scrapowania (sub-repozytorium).

## Przygotowanie danych (src/data)

Warstwa `src/data` przygotowuje dane do dalszych krokow.

Przykład uzycia w notatniku:

    from src.data import load_raw_seasons, add_odds_columns

    raw_df = load_raw_seasons("data/raw/1liga_*.json")
    df = add_odds_columns(raw_df, trim_drop=1)

lub:

    from src.data import load_and_add_odds_columns
    df = load_and_add_odds_columns(pattern="data/raw/1liga_*.json", trim_drop=1)

wersja compact (usuwa surowe kolumny marketow po wyliczeniu kursow):

    from src.data import load_and_add_odds_columns_compact
    df = load_and_add_odds_columns_compact(pattern="data/raw/1liga_*.json", trim_drop=1)
