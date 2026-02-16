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