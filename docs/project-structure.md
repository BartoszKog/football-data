---
title: Struktura projektu
summary: Gdzie leżą dane, kod źródłowy, notatniki i skrypty — mapa katalogów football-data
sidebar_title: Struktura projektu
order: 2
description: Przegląd organizacji repozytorium football-data — data, src, notebooks, scripts, dokumentacja MkDocs.
keywords: football-data, struktura projektu, src, data, notebooks, mkdocs
---

# +lucide:folder-tree+ Struktura projektu

Ta strona to **mapa repozytorium**: po co jest który katalog i jak to łączy się z
pipeline’em od surowych plików po modele. Szczegóły instalacji i pierwszych
kroków są w [Getting started](getting-started.md); **krok po kroku** (ładowanie,
cechy, modele, ewaluacja, tuning) — w [Guides](guides/01-loading-data.md);
krótki opis repozytorium dla GitHub — w **`README.md`** w korzeniu.

## +lucide:folder+ Drzewo katalogów

```text
football-data/
├── README.md              # Krótki opis + link do dokumentacji MkDocs
├── pyproject.toml         # Zależności i metadane pakietu (uv)
├── mkdocs.yml             # Konfiguracja dokumentacji MkDocs
├── overrides/             # Szablony motywu (linki działają też pod file://)
│
├── docs/                  # Źródła tej dokumentacji (.md, assets)
├── site/                  # Wynik `mkdocs build` (lokalnie; w .gitignore)
│
├── data/
│   ├── external/          # Słowniki i dane pomocnicze z zewnątrz
│   ├── raw/               # Surowe, nieedytowane pliki (np. JSON z OddsHarvester)
│   └── processed/         # Oczyszczone tabele pod modelowanie (skrypty / notatniki)
│
├── notebooks/
│   ├── exploration/       # Eksperymenty, brudnopisy, szybkie prototypy
│   ├── features_lab/      # Eksperymenty z cechami
│   └── reports/           # „Czyste” raporty (np. pod publikację wyników)
│
├── outputs/
│   ├── figures/           # Wykresy do raportów (png, svg)
│   └── reports/           # Eksport HTML/PDF itd.
│
├── sandbox/               # Kod tymczasowy (ignorowany przez git)
├── scripts/               # Skrypty CLI: batch scrapingu, retry, generowanie figurek
├── tests/                 # Testy pytest (m.in. splits i grid search)
│
├── src/                   # Biblioteka Python — warstwy data → features → models
│   ├── data/              # Wczytywanie sezonów, kolumny kursowe
│   ├── features/          # Prawdopodobieństwa z kursów, lambdy Poissona
│   ├── visualization/     # Wspólne wykresy (na razie puste; pod przyszły kod)
│   └── models/            # Interfejsy, modele, ewaluacja, tuning, komponenty
│       ├── interfaces.py
│       ├── statistical/   # Poisson + Dixon–Coles
│       ├── ml/            # XGBoost Poisson
│       ├── evaluation/    # Punktacja, deviance, wykresy podsumowań
│       ├── tuning/        # Grid search, walk-forward
│       └── components/    # Budowa macierzy scoreline, optymalizacja
│
└── OddsHarvester/         # Osobne narzędzie do scrapowania (submoduł / klon; w .gitignore)
```

## +lucide:database+ Warstwa danych (`data/`)

- **`raw/`** — wejście z harvestera: nie modyfikuj ręcznie; traktuj jako źródło prawdy.
- **`processed/`** — wynik ETL i featurek: to zwykle wczytujesz w notatnikach pod trening.
- **`external/`** — statyczne słowniki lub pliki referencyjne spoza scrapera.

Dane nie powinny trafiać do gita (wzorce w `.gitignore`); trzymasz je lokalnie lub
w swoim backupie.

## +lucide:package+ Kod biblioteczny (`src/`)

Projekt trzyma się podziału: **dane → cechy → modele**. Importujesz z pakietu
`src` (środowisko ustawione przez `uv sync` — patrz [Getting started](getting-started.md)).

- **`src.data`** — wczytywanie surowych sezonów, budowa kolumn kursowych
  (`trimmed_avg_*`, `max_*`, …). API: [src.data](api/data.md).
- **`src.features`** — prawdopodobieństwa implikowane z kursów, lambdy baseline /
  skalibrowane. API: [src.features](api/features.md).
- **`src.models`** — kontrakty `PredictiveModel` / `TrainablePredictiveModel`,
  implementacje (`PoissonDixonColesModel`, `XGBoostPoissonModel`).
  API: [src.models](api/models.md).
- **`src.models.evaluation`** — punkty zgodne z Supertyperem, Poisson deviance,
  wykresy typu „predictions summary”. API: [src.models.evaluation](api/evaluation.md).
- **`src.models.tuning`** — grid search (predictive i trainable), sezonowy
  walk-forward train/val/eval. API: [src.models.tuning](api/tuning.md).
- **`src.visualization`** — pod wspólne helpery wizualizacji; katalog jest w repo,
  moduł wypełnisz w kolejnych iteracjach.

## +lucide:notebook-text+ Notatniki (`notebooks/`)

- **`exploration/`** — miejsce na ścieżki „co by było, gdyby”, porównania modeli, szybkie eksperymenty.
- **`features_lab/`** — izolowane eksperymenty z cechami i transformacjami kursów.
- **`reports/`** — krótsze, czytelne notatniki (np. Marimo) prezentujące ustalone wnioski.

Katalogi sesji `__marimo__/` są artefaktem edytora — zwykle ignorowane w gicie.

## +lucide:image+ Wyniki (`outputs/`)

Trzymaj tu **wygenerowane** pliki: wykresy do dokumentacji lub raportów, eksporty
HTML, żeby nie zaśmiecać `notebooks/` i żeby łatwo je podlinkować z raportów.

## +lucide:terminal+ Skrypty (`scripts/`)

Jednorazowe i automatyzacje uruchamiane z CLI: batch scrapingu, ponawianie
nieudanych meczów, generowanie figurek do docs. Nie zastępują one API z `src/` —
to raczej „klejenie” kroków pod konkretne zadanie operacyjne.

## +lucide:flask-conical+ Testy (`tests/`)

Regresje dla logiki czasu (walk-forward) i grid searcha — uruchamiasz przez pytest
w środowisku projektu (`uv run pytest`).

## +lucide:book-open+ Dokumentacja (`docs/` i `mkdocs.yml`)

- **`docs/*.md`** — strony z front-matterem (nawigacja z folderów + kolejność z `order:`).
- **`docs/assets/`** — obrazki do figure-caption w treści.
- **`mkdocs.yml`** — motyw `mkdocs-shadcn`, rozszerzenia Markdown, mkdocstrings.
- **`overrides/`** — szablony tak, by linki działały także przy otwieraniu
  `site/*.html` z dysku (`file://`).

Katalog **`site/`** powstaje po `uv run mkdocs build` — to wyjście buildu, nie pliki do ręcznej edycji.

## +lucide:box+ Sandbox i narzędzia zewnętrzne

- **`sandbox/`** — dowolny kod pomocniczy; domyślnie poza repozytorium w sensie
  wersjonowania (`.gitignore`).
- **`OddsHarvester/`** — osobne repozytorium ze scraperem; klonujesz obok / jako
  submoduł według instrukcji w [Getting started](getting-started.md).

!!! tip "Gdzie szukać przykładów kodu?"
    Ta strona jest **orientacyjna**. Pełne snippet’y pipeline’u są w
    **[Guides](guides/01-loading-data.md)**; sygnatury i docstringi — w
    **[API](api/data.md)**; `README.md` w korzeniu trzyma skrót i link do
    `mkdocs serve` / build.
