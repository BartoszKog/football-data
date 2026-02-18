from __future__ import annotations

import ast
from typing import Any, Iterable, Mapping

import pandas as pd


def _to_float(value: Any) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _normalize_market_list(market: Any) -> list[dict[str, Any]]:
    if isinstance(market, list):
        return [item for item in market if isinstance(item, dict)]
    if isinstance(market, str):
        try:
            parsed = ast.literal_eval(market)
            if isinstance(parsed, list):
                return [item for item in parsed if isinstance(item, dict)]
        except (SyntaxError, ValueError):
            return []
    return []


def _clean_values(values: Iterable[Any]) -> list[float]:
    cleaned: list[float] = []
    for value in values:
        as_float = _to_float(value)
        if as_float is not None:
            cleaned.append(as_float)
    return cleaned


def safe_max(values: Iterable[Any]) -> float | None:
    """Return max numeric value or None if no valid numbers."""
    cleaned = _clean_values(values)
    return max(cleaned) if cleaned else None


def safe_mean(values: Iterable[Any]) -> float | None:
    """Return arithmetic mean or None if no valid numbers."""
    cleaned = _clean_values(values)
    if not cleaned:
        return None
    return float(sum(cleaned) / len(cleaned))


def trimmed_mean(values: Iterable[Any], drop: int = 1) -> float | None:
    """
    Return trimmed mean after dropping `drop` lowest and highest values.

    If there are too few values for trimming, this falls back to plain mean.
    """
    cleaned = sorted(_clean_values(values))
    if not cleaned:
        return None
    if drop <= 0 or len(cleaned) <= 2 * drop:
        return float(sum(cleaned) / len(cleaned))
    trimmed = cleaned[drop:-drop]
    if not trimmed:
        return float(sum(cleaned) / len(cleaned))
    return float(sum(trimmed) / len(trimmed))


def aggregate_market_odds(
    market: Any,
    selections: Mapping[str, str],
    drop: int = 1,
) -> dict[str, float | None]:
    """
    Aggregate odds for a market with three metrics:
    - max_<selection>
    - avg_<selection>
    - trimmed_avg_<selection>
    """
    market_items = _normalize_market_list(market)
    output: dict[str, float | None] = {}

    for out_name, source_key in selections.items():
        values = [book.get(source_key) for book in market_items]
        output[f"max_{out_name}"] = safe_max(values)
        output[f"avg_{out_name}"] = safe_mean(values)
        output[f"trimmed_avg_{out_name}"] = trimmed_mean(values, drop=drop)

    return output


def add_market_features(
    df: pd.DataFrame,
    market_column: str,
    selections: Mapping[str, str],
    drop: int = 1,
) -> pd.DataFrame:
    """
    Return a copy of DataFrame with odds features for one market column.

    Added columns for each selection:
    - `max_<name>`
    - `avg_<name>`
    - `trimmed_avg_<name>`
    """
    if market_column not in df.columns:
        return df.copy()

    features = df[market_column].apply(
        lambda market: aggregate_market_odds(market, selections=selections, drop=drop)
    )
    features_df = pd.DataFrame(features.tolist(), index=df.index)
    return pd.concat([df.copy(), features_df], axis=1)

