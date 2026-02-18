from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


SEASON_FILE_RE = re.compile(r"1liga_(\d{4})(\d{4})\.json$", re.IGNORECASE)


def _extract_season_from_filename(path: Path) -> str:
    """Extract season label from a raw file name."""
    match = SEASON_FILE_RE.search(path.name)
    if match:
        start, end = match.groups()
        return f"{start}/{end}"
    if path.stem.endswith("_current"):
        return "current"
    return path.stem


def _read_json_file(path: Path) -> list[dict[str, Any]]:
    """Read a raw JSON file and always return a list of row dicts."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        return [data]
    return []


def load_raw_seasons(pattern: str = "data/raw/1liga_*.json") -> pd.DataFrame:
    """
    Load and merge many season JSON files into one DataFrame.

    Responsibilities:
    - read all files matching `pattern`,
    - append metadata columns: `season` and `source_file`,
    - normalize date and score types,
    - deduplicate rows by `match_link`:
      keep the row with latest `scraped_date` and fallback to last row order,
    - print a short info message when duplicates are removed.

    Notes:
    - `match_date` is converted to timezone-aware `Europe/Warsaw`,
    - `scraped_date` stays timezone-aware UTC.
    """
    files = sorted(Path(".").glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")

    frames: list[pd.DataFrame] = []
    for file_path in files:
        rows = _read_json_file(file_path)
        if not rows:
            continue

        frame = pd.DataFrame(rows)
        frame["season"] = _extract_season_from_filename(file_path)
        frame["source_file"] = file_path.name
        frames.append(frame)

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)

    if "match_date" in df.columns:
        df["match_date"] = (
            pd.to_datetime(df["match_date"], errors="coerce", utc=True)
            .dt.tz_convert("Europe/Warsaw")
        )
    if "scraped_date" in df.columns:
        df["scraped_date"] = pd.to_datetime(df["scraped_date"], errors="coerce", utc=True)
    for score_col in ("home_score", "away_score"):
        if score_col in df.columns:
            df[score_col] = pd.to_numeric(df[score_col], errors="coerce")

    if "match_link" in df.columns:
        before_count = len(df)
        df = df.copy()
        df["_row_order"] = range(len(df))

        if "scraped_date" in df.columns:
            df = df.sort_values(
                by=["match_link", "scraped_date", "_row_order"],
                ascending=[True, True, True],
                na_position="first",
            )
        else:
            df = df.sort_values(by=["match_link", "_row_order"], ascending=[True, True])

        df = df.drop_duplicates(subset=["match_link"], keep="last")
        removed_count = before_count - len(df)
        if removed_count > 0:
            print(
                "[load_raw_seasons] Removed "
                f"{removed_count} duplicate rows based on match_link."
            )

        df = (
            df.sort_values("_row_order")
            .drop(columns=["_row_order"])
            .reset_index(drop=True)
        )

    return df

