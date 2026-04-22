---
title: Getting started
summary: Od zera do pierwszego notatnika i lokalnego buildu dokumentacji
sidebar_title: Getting started
order: 1
new: true
description: Instalacja projektu football-data, OddsHarvester, pobranie danych, pierwszy notatnik i build dokumentacji.
keywords: football-data, getting-started, uv, marimo, mkdocs, oddsharvester
---

# +lucide:rocket+ Getting started

Odtworzenie projektu od zera do uruchomienia pierwszego notatnika i lokalnego
buildu dokumentacji.

## +lucide:clipboard-check+ Wymagania

- Python ≥ 3.12
- [uv](https://github.com/astral-sh/uv) jako menedżer środowiska
- Git (dla głównego repo; OddsHarvester klonujesz osobno)

## +lucide:git-branch+ 1. Sklonuj repo i przygotuj środowisko

```bash
git clone <url-repo> football-data
cd football-data
uv sync
```

`uv sync` instaluje wszystkie runtime deps z `pyproject.toml` (pandas, scipy,
xgboost, pygam, marimo itd.) zgodnie z `uv.lock`.

Aby dodatkowo zainstalować narzędzia do pracy nad dokumentacją:

```bash
uv sync --group dev
```

Zainstaluje `mkdocs`, `mkdocs-shadcn`, `mkdocstrings[python]`,
`pymdown-extensions`, `Pygments` oraz `ruff`.

!!! note "numpy override"
    `pyproject.toml` zawiera `[tool.uv] override-dependencies = ["numpy>=2.0"]`
    — potrzebne bo `pygam==0.9.0` w metadanych ma `numpy<2.0`, ale w praktyce
    działa z numpy 2.x (potwierdzone eksperymentalnie w
    `notebooks/exploration/02_GAM_Lab.py`, gdzie dodatkowo patchowany jest
    `scipy.sparse.*.A`).

## +lucide:download+ 2. Pobierz i przygotuj OddsHarvester

`OddsHarvester/` **nie jest commitowany** do repo (wpis w `.gitignore` linia 13).
To zewnętrzny scraper z własnym `uv`-owym środowiskiem.

```bash
git clone <url-oddsharvester> OddsHarvester
cd OddsHarvester
uv sync
cd ..
```

Po tym kroku struktura powinna wyglądać tak:

```text
football-data/
├── OddsHarvester/           # sklonowany scraper z własnym .venv
│   ├── .venv/
│   └── ...
└── (reszta projektu)
```

`scripts/scrape_seasons_batch.py` oczekuje, że python OddsHarvestera jest pod
`OddsHarvester/.venv/Scripts/python.exe` (Windows).

## +lucide:database+ 3. Pobierz surowe dane

Katalogi `data/raw/`, `data/processed/`, `data/external/` są **nie-śledzone**
przez git (patrz `.gitignore` linie 22-28) — dane musisz wygenerować lokalnie.

### Pobranie sezonów 1. Ligi

Edytuj `scripts/scrape_seasons_batch.py`, ustaw listę `SEASONS_TO_SCRAPE`
(np. `["2023-2024", "2022-2023", "2021-2022", "2020-2021", "current"]`) i uruchom:

```bash
uv run python scripts/scrape_seasons_batch.py
```

Rezultat: pliki `data/raw/<nazwa-danych>_<sezon>.json` + logi w `logs/`.

### Pojedyncze ponawianie

Jeśli pojedyncze mecze padły przy scrapowaniu:

```bash
uv run python scripts/retry_failed_matches.py
uv run python scripts/manual_retry_matches.py
```

## +lucide:flask-conical+ 4. Uruchom pierwszy notatnik

```bash
uv run marimo edit notebooks/reports/poisson_dixon_coles_baseline.py
```

Marimo otworzy przeglądarkę z interaktywnym notatnikiem. Powinien wczytać dane
z `data/raw/` i policzyć baseline Poisson Dixon-Coles.

!!! tip "Konwencja notatników"
    - `notebooks/reports/` — finalne, "wyczyszczone" analizy (też eksportowane
      do HTML do `outputs/reports/`).
    - `notebooks/exploration/` — brudnopisy, eksperymenty, sprawdzanie
      pomysłów (01_xgboost, 02_GAM, 03_Dixon_Coles).
    - `notebooks/features_lab/` — eksperymenty z cechami.

## +lucide:book-open+ 5. Uruchom dokumentację offline

Projekt używa `mkdocs` + motywu `mkdocs-shadcn`.

### Tryb live (rekomendowany podczas edycji docs)

```bash
uv run mkdocs serve
```

Dostępne pod `http://127.0.0.1:8000`. Auto-reload przy zapisie pliku `.md`.

### Build statyczny (do otwierania z dysku)

```bash
uv run mkdocs build
```

Generuje folder `site/` (nie-śledzony przez git — wpis w `.gitignore`). W
`mkdocs.yml` ustawione jest `use_directory_urls: false`, dzięki czemu każda
strona to osobny plik `.html` (np. `site/getting-started.html`,
`site/api/data.html`). Otwórz `site/index.html` dwuklikiem — wewnętrzne
linki działają spod `file://` bez serwera.

## +lucide:check-check+ Szybka checklist

- `uv sync` bez błędów
- `uv sync --group dev` bez błędów
- `OddsHarvester/` sklonowany i jego `uv sync` przeszedł
- `data/raw/1liga*.json` ma pliki (przynajmniej 1 sezon)
- `uv run marimo edit notebooks/reports/poisson_dixon_coles_baseline.py` otwiera notatnik
- `uv run mkdocs build` kończy bez błędów i `site/index.html` da się otworzyć
