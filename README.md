# Football Data

[Dokumentacja](https://bartoszkog.github.io/football-data/index.html)

Osobisty projekt pod modele typu Poisson / Dixon–Coles i XGBoost Poisson do
typowania wyników (m.in. [Supertyper](https://www.app-helper.com/betting_game/?appid=50832)).

## Dokumentacja

**[Dokumentacja online (GitHub Pages)](https://bartoszkog.github.io/football-data/index.html)** —
przewodniki, API, koncepty. Pełny pipeline kodu (dane → cechy → modele →
ewaluacja → tuning) jest w sekcji [Guides](https://bartoszkog.github.io/football-data/guides/01-loading-data.html).

| | |
|---|---|
| Lokalnie (edycja) | `uv sync --group dev` → `uv run mkdocs serve` → http://127.0.0.1:8000 |
| Build statyczny | `uv run mkdocs build` → folder `site/` |
| W dokumentacji | [Strona główna](https://bartoszkog.github.io/football-data/index.html) · [Getting started](https://bartoszkog.github.io/football-data/getting-started.html) |

### Raport baseline Poisson Dixon–Coles

Badawczy baseline `PoissonDixonColesModel`: lambdy z kursów 1X2 i Over/Under 2.5,
grid search po `rho` i `bias_correction`, porównanie metryk konkursowych (`avg_points`,
`NLL`) z diagnostyką PIT oraz testem na aktualnej próbie Supertypera. To diagnoza
modelu opartego wyłącznie na kursach — nie gotowa procedura typowania.

Statyczny HTML z notatnika
[`notebooks/reports/poisson_dixon_coles_baseline.py`](notebooks/reports/poisson_dixon_coles_baseline.py):

- **[Pełny raport (HTML)](https://bartoszkog.github.io/football-data/reports/poisson_dixon_coles_baseline.html)**
- [Skrót, wnioski i mapa sekcji](https://bartoszkog.github.io/football-data/reports/poisson-dixon-coles-baseline.html) —
  strona w dokumentacji

## Organizacja projektu

```text
football-data/
├── README.md             <- Ten plik + link do docs/
├── docs/                 <- Źródła dokumentacji MkDocs
├── data/
│   ├── external/
│   ├── raw/              <- JSON-y z OddsHarvester
│   └── processed/
├── notebooks/
│   ├── exploration/
│   ├── features_lab/
│   └── reports/
├── outputs/              <- Wykresy, raporty
├── sandbox/
├── scripts/
├── src/                  <- data → features → models
├── tests/
└── OddsHarvester/        <- Osobny klon scrapera (w .gitignore)
```

## Szybki start

```bash
git clone https://github.com/BartoszKog/football-data.git football-data
cd football-data
uv sync
```

Dane, OddsHarvester i pierwszy notatnik: [Getting started](https://bartoszkog.github.io/football-data/getting-started.html).

Minimalny import (po przygotowaniu `df` z kursami i prawdopodobieństwami):

```python
from src.models import PoissonDixonColesModel

pred_df = PoissonDixonColesModel(rho=-0.06).predict(df)
```

Więcej przykładów: [Guides](https://bartoszkog.github.io/football-data/guides/01-loading-data.html) i [API](https://bartoszkog.github.io/football-data/api/data.html).
