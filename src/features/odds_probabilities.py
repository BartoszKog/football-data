"""Feature helpers for converting odds into implied probabilities."""

from __future__ import annotations

from typing import Any, Literal, Sequence

import numpy as np
import pandas as pd
from scipy import optimize

_STANDARD_MARKETS: dict[str, tuple[str, ...]] = {
    "1x2": ("1", "X", "2"),
    "btts": ("btts_yes", "btts_no"),
    "over_under_25": ("over_25", "under_25"),
}


def _power_implied_probabilities_from_odds(
    odds: Sequence[Any],
    *,
    min_odds: float = 1.0,
    initial_k: float = 1.0,
    max_iter: int = 50,
    tolerance: float = 1e-8,
) -> np.ndarray:
    """Return implied probabilities for one odds vector using power method."""
    odds_array = np.asarray(odds, dtype=float)

    if odds_array.ndim != 1 or odds_array.size == 0:
        raise ValueError("odds must be a non-empty 1D sequence.")
    if not np.all(np.isfinite(odds_array)):
        raise ValueError("odds contain non-finite values.")
    if np.any(odds_array <= min_odds):
        raise ValueError(f"odds must be greater than min_odds={min_odds}.")

    raw_probs = 1.0 / odds_array
    raw_sum = float(raw_probs.sum())
    if np.isclose(raw_sum, 1.0, atol=tolerance, rtol=0.0):
        return raw_probs

    def objective(k: float) -> float:
        return float(np.sum(raw_probs**k) - 1.0)

    k_opt = optimize.newton(objective, x0=initial_k, maxiter=max_iter)
    final_probs = raw_probs**k_opt

    if not np.all(np.isfinite(final_probs)):
        raise RuntimeError("power-method result is non-finite.")

    return final_probs


def add_power_implied_probabilities(
    df: pd.DataFrame,
    odds_columns: Sequence[str],
    output_columns: Sequence[str] | None = None,
    *,
    min_odds: float = 1.0,
    initial_k: float = 1.0,
    max_iter: int = 50,
    tolerance: float = 1e-8,
    errors: Literal["coerce", "raise"] = "coerce",
) -> pd.DataFrame:
    """
    Add implied-probability columns computed with the power method.

    The method solves `sum((1/odds)^k) = 1` for each row and uses the resulting
    exponent to remove bookmaker margin from raw implied probabilities.

    Parameters
    ----------
    df:
        Input DataFrame containing odds columns.
    odds_columns:
        Column names with odds for one market (2-way, 3-way, n-way).
    output_columns:
        Output probability column names. If None, names are generated as
        `prob_<input_column_name>`.
    min_odds:
        Minimal allowed odds value. Each odds value must be greater than this.
    initial_k:
        Initial exponent value for Newton solver.
    max_iter:
        Maximum number of Newton iterations.
    tolerance:
        Absolute tolerance used for checking whether raw implied probabilities
        already sum to 1.
    errors:
        Error policy for row-level failures:
        - ``"coerce"`` fills output columns with ``NaN`` for failed rows,
        - ``"raise"`` raises ``ValueError`` with row index context.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added probability columns.
    """
    if errors not in {"coerce", "raise"}:
        raise ValueError("errors must be either 'coerce' or 'raise'.")

    odds_cols = list(odds_columns)
    if not odds_cols:
        raise ValueError("odds_columns cannot be empty.")

    missing_cols = [col for col in odds_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing odds columns in DataFrame: {missing_cols}")

    if output_columns is None:
        out_cols = [f"prob_{col}" for col in odds_cols]
    else:
        out_cols = list(output_columns)
        if len(out_cols) != len(odds_cols):
            raise ValueError(
                "output_columns must have the same length as odds_columns."
            )

    results: list[np.ndarray] = []
    for row_idx, odds_row in df[odds_cols].iterrows():
        try:
            row_probs = _power_implied_probabilities_from_odds(
                odds=odds_row.to_numpy(dtype=float),
                min_odds=min_odds,
                initial_k=initial_k,
                max_iter=max_iter,
                tolerance=tolerance,
            )
        except Exception as exc:  # pragma: no cover - narrowed by error mode
            if errors == "raise":
                raise ValueError(
                    f"Failed to compute probabilities for row {row_idx}: {exc}"
                ) from exc
            row_probs = np.full(len(odds_cols), np.nan, dtype=float)
        results.append(row_probs)

    probs_df = pd.DataFrame(results, columns=out_cols, index=df.index, dtype=float)
    return pd.concat([df.copy(), probs_df], axis=1)


def add_power_implied_probabilities_standard_markets(
    df: pd.DataFrame,
    *,
    odds_prefix: Literal["max", "avg", "trimmed_avg"] = "trimmed_avg",
    output_prefix: str = "prob",
    min_odds: float = 1.0,
    initial_k: float = 1.0,
    max_iter: int = 50,
    tolerance: float = 1e-8,
    errors: Literal["coerce", "raise"] = "coerce",
    missing_markets: Literal["raise", "skip"] = "raise",
) -> pd.DataFrame:
    """
    Add power-method implied probabilities for default football markets.

    This convenience wrapper computes probabilities for:
    - ``1x2``: ``1``, ``X``, ``2``,
    - ``btts``: ``btts_yes``, ``btts_no``,
    - ``over_under_25``: ``over_25``, ``under_25``.

    Input odds columns are selected from one prefix chosen by ``odds_prefix``
    (``max``, ``avg`` or ``trimmed_avg``). Output columns use ``output_prefix``
    and keep market suffixes, e.g. ``prob_1``, ``prob_X``, ``prob_btts_yes``.

    Parameters
    ----------
    df:
        Input DataFrame containing odds columns prepared in data layer.
    odds_prefix:
        Prefix used to pick source odds columns.
    output_prefix:
        Prefix for generated probability columns.
    min_odds:
        Minimal allowed odds value. Each odds value must be greater than this.
    initial_k:
        Initial exponent value for Newton solver.
    max_iter:
        Maximum number of Newton iterations.
    tolerance:
        Absolute tolerance used for checking whether raw implied probabilities
        already sum to 1.
    errors:
        Error policy for row-level failures forwarded to
        ``add_power_implied_probabilities``.
    missing_markets:
        - ``"raise"`` raises if any required market column is missing,
        - ``"skip"`` skips markets with missing columns.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added probability columns.
    """
    if missing_markets not in {"raise", "skip"}:
        raise ValueError("missing_markets must be either 'raise' or 'skip'.")
    if not output_prefix:
        raise ValueError("output_prefix cannot be empty.")

    result = df.copy()
    for market_name, suffixes in _STANDARD_MARKETS.items():
        odds_columns = [f"{odds_prefix}_{suffix}" for suffix in suffixes]
        missing_cols = [col for col in odds_columns if col not in result.columns]
        if missing_cols:
            if missing_markets == "raise":
                raise ValueError(
                    f"Missing columns for market '{market_name}': {missing_cols}"
                )
            continue

        output_columns = [f"{output_prefix}_{suffix}" for suffix in suffixes]
        result = add_power_implied_probabilities(
            result,
            odds_columns=odds_columns,
            output_columns=output_columns,
            min_odds=min_odds,
            initial_k=initial_k,
            max_iter=max_iter,
            tolerance=tolerance,
            errors=errors,
        )

    return result
