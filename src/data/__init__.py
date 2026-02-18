"""Data preparation package for football match datasets.

This package provides notebook-friendly helpers to:
- load and merge raw season JSON files,
- deduplicate matches by `match_link`,
- normalize `match_date` to `Europe/Warsaw`,
- compute odds aggregates (`max_*`, `avg_*`, `trimmed_avg_*`).

Public API:
- load_raw_seasons
- add_odds_columns
- add_odds_columns_compact
- load_and_add_odds_columns
- load_and_add_odds_columns_compact
"""

from .io import load_raw_seasons
from .prepare_matches import (
    add_odds_columns,
    add_odds_columns_compact,
    load_and_add_odds_columns,
    load_and_add_odds_columns_compact,
)

__all__ = [
    "load_raw_seasons",
    "add_odds_columns",
    "add_odds_columns_compact",
    "load_and_add_odds_columns",
    "load_and_add_odds_columns_compact",
]

