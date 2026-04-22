# Football Data

Osobisty projekt pod modele typu Poisson / Dixon–Coles i XGBoost Poisson do
typowania wyników (m.in. [Supertyper](https://www.app-helper.com/betting_game/?appid=50832)).

**Dokumentacja (MkDocs):** po sklonowaniu repo uruchom `uv sync --group dev`, potem:

```bash
uv run mkdocs serve
```

Build statyczny: `uv run mkdocs build` — wynik w `site/` (szczegóły w
[Getting started](docs/getting-started.md)). Pełny pipeline kodu (dane → cechy →
modele → ewaluacja → tuning) jest w folderze **`docs/guides/`**.

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
git clone <url-repo> football-data
cd football-data
uv sync
```

Dane, OddsHarvester i pierwszy notatnik: [docs/getting-started.md](docs/getting-started.md).

Minimalny import (po przygotowaniu `df` z kursami i prawdopodobieństwami):

```python
from src.models import PoissonDixonColesModel

pred_df = PoissonDixonColesModel(rho=-0.06).predict(df)
```

Więcej przykładów: [Guides](docs/guides/01-loading-data.md) i [API](docs/api/data.md).
