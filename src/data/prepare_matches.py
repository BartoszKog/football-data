from __future__ import annotations

import pandas as pd

from .io import load_raw_seasons
from .odds_features import add_market_features


MARKET_CONFIG = {
    "1x2_market": {"1": "1", "X": "X", "2": "2"},
    "btts_market": {"btts_yes": "btts_yes", "btts_no": "btts_no"},
    "over_under_2_5_market": {"over_25": "odds_over", "under_25": "odds_under"},
}


def add_odds_columns(
    matches_df: pd.DataFrame,
    trim_drop: int = 1,
    sort_by_date: bool = True,
    drop_source_market_columns: bool = False,
) -> pd.DataFrame:
    """
    Compute odds-based feature columns for match rows.

    This function focuses only on data-layer transformations:
    - uses rows prepared by the data loader (`load_raw_seasons`),
    - coerces `home_score` and `away_score` to numeric,
    - adds odds aggregate columns for supported markets:
      `max_*`, `avg_*`, `trimmed_avg_*`,
    - optionally sorts by `match_date`,
    - optionally drops source market columns after feature extraction.

    Parameters
    ----------
    matches_df:
        Input DataFrame containing raw match rows.
    trim_drop:
        Number of lowest and highest odds values dropped in trimmed mean.
    sort_by_date:
        If True, sort output by `match_date`.
    drop_source_market_columns:
        If True, remove source market columns used to compute odds features:
        `1x2_market`, `btts_market`, `over_under_2_5_market`.

    Returns
    -------
    pd.DataFrame
        DataFrame with added odds feature columns.
    """
    df = matches_df.copy()

    for score_col in ("home_score", "away_score"):
        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    for market_column, selections in MARKET_CONFIG.items():
        df = add_market_features(
            df=df,
            market_column=market_column,
            selections=selections,
            drop=trim_drop,
        )

    if drop_source_market_columns:
        market_cols_to_drop = [col for col in MARKET_CONFIG if col in df.columns]
        if market_cols_to_drop:
            df = df.drop(columns=market_cols_to_drop)

    if sort_by_date and "match_date" in df.columns:
        df = df.sort_values("match_date").reset_index(drop=True)

    return df


def add_odds_columns_compact(
    matches_df: pd.DataFrame,
    trim_drop: int = 1,
    sort_by_date: bool = True,
) -> pd.DataFrame:
    """
    Add odds feature columns and drop raw market payload columns.

    This is a compact convenience wrapper over `add_odds_columns` with
    `drop_source_market_columns=True`.

    Removed source columns (if present):
    - `1x2_market`
    - `btts_market`
    - `over_under_2_5_market`

    Parameters
    ----------
    matches_df:
        Input DataFrame containing match rows.
    trim_drop:
        Number of lowest and highest odds values dropped in trimmed mean.
    sort_by_date:
        If True, sort output by `match_date`.

    Returns
    -------
    pd.DataFrame
        DataFrame with odds features and without raw market columns.
    """
    return add_odds_columns(
        matches_df=matches_df,
        trim_drop=trim_drop,
        sort_by_date=sort_by_date,
        drop_source_market_columns=True,
    )


def load_and_add_odds_columns(
    pattern: str = "data/raw/1liga_*.json",
    trim_drop: int = 1,
    sort_by_date: bool = True,
    drop_source_market_columns: bool = False,
) -> pd.DataFrame:
    """
    Load raw season files and add odds feature columns.

    Loading step (`load_raw_seasons`) also performs:
    - season/source metadata columns,
    - deduplication by `match_link` (newest `scraped_date` wins),
    - `match_date` timezone normalization to `Europe/Warsaw`,
    - informational `print` when duplicates are removed.

    Then this function computes odds columns:
    - `max_*`
    - `avg_*`
    - `trimmed_avg_*`

    Parameters
    ----------
    pattern:
        Glob for raw season files, e.g. `data/raw/1liga_*.json`.
    trim_drop:
        Number of lowest and highest odds values dropped in trimmed mean.
    sort_by_date:
        If True, sort output by `match_date`.
    drop_source_market_columns:
        If True, remove raw market payload columns:
        `1x2_market`, `btts_market`, `over_under_2_5_market`.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame with odds feature columns.
    """
    matches_df = load_raw_seasons(pattern=pattern)
    return add_odds_columns(
        matches_df=matches_df,
        trim_drop=trim_drop,
        sort_by_date=sort_by_date,
        drop_source_market_columns=drop_source_market_columns,
    )


def load_and_add_odds_columns_compact(
    pattern: str = "data/raw/1liga_*.json",
    trim_drop: int = 1,
    sort_by_date: bool = True,
) -> pd.DataFrame:
    """
    Load raw season files, add odds columns, and remove raw market columns.

    This is a compact convenience wrapper over `load_and_add_odds_columns`
    with `drop_source_market_columns=True`.

    Removed source columns (if present):
    - `1x2_market`
    - `btts_market`
    - `over_under_2_5_market`

    Parameters
    ----------
    pattern:
        Glob for raw season files, e.g. `data/raw/1liga_*.json`.
    trim_drop:
        Number of lowest and highest odds values dropped in trimmed mean.
    sort_by_date:
        If True, sort output by `match_date`.

    Returns
    -------
    pd.DataFrame
        Prepared DataFrame with odds features and without raw market columns.
    """
    return load_and_add_odds_columns(
        pattern=pattern,
        trim_drop=trim_drop,
        sort_by_date=sort_by_date,
        drop_source_market_columns=True,
    )

