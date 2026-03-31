"""Feature helpers for baseline Poisson goal-rate priors."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy.stats import poisson


def add_baseline_poisson_lambdas(
    df: pd.DataFrame,
    *,
    prob_home_col: str = "prob_trimmed_avg_1",
    prob_away_col: str = "prob_trimmed_avg_2",
    prob_over25_col: str = "prob_trimmed_avg_over_25",
    output_home_col: str = "baseline_lambda_home",
    output_away_col: str = "baseline_lambda_away",
    bias_correction: float = 1.035,
    lambda_min: float = 0.01,
    lambda_max: float = 20.0,
    grid_size: int = 10_000,
    errors: Literal["coerce", "raise"] = "coerce",
) -> pd.DataFrame:
    """
    Add baseline Poisson lambdas derived from 1X2 and over-2.5 probabilities.

    The function maps ``prob_over25`` to total expected goals via interpolation
    on a Poisson grid and then splits total lambda between home/away sides using
    ``prob_home`` and ``prob_away`` shares.

    Parameters
    ----------
    df:
        Input DataFrame containing probability columns used to derive lambdas.
    prob_home_col:
        Column with home-win probability proxy (used for home share of total
        lambda), e.g. ``prob_trimmed_avg_1``.
    prob_away_col:
        Column with away-win probability proxy (used for away share of total
        lambda), e.g. ``prob_trimmed_avg_2``.
    prob_over25_col:
        Column with probability of total goals over 2.5. This value is mapped to
        total expected goals via interpolation on a Poisson CDF grid.
    output_home_col:
        Name of output column where home baseline lambda will be stored.
    output_away_col:
        Name of output column where away baseline lambda will be stored.
    bias_correction:
        Multiplicative correction applied to interpolated total lambda.
        Must be finite and greater than 0.
    lambda_min:
        Lower bound of interpolation grid for total lambda. Must be > 0.
    lambda_max:
        Upper bound of interpolation grid for total lambda.
        Must be greater than ``lambda_min``.
    grid_size:
        Number of points used to build interpolation grid
        (``linspace(lambda_min, lambda_max, grid_size)``). Must be >= 100.
    errors:
        Row-level error policy:
        - ``"coerce"`` sets invalid outputs to ``NaN``,
        - ``"raise"`` raises ``ValueError`` with example row indices.

    Returns
    -------
    pd.DataFrame
        Copy of input DataFrame with added baseline lambda columns:
        ``output_home_col`` and ``output_away_col``.
    """
    if errors not in {"coerce", "raise"}:
        raise ValueError("errors must be either 'coerce' or 'raise'.")

    required_cols = [prob_home_col, prob_away_col, prob_over25_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing probability columns in DataFrame: {missing_cols}")

    if not np.isfinite(bias_correction) or bias_correction <= 0:
        raise ValueError("bias_correction must be finite and greater than 0.")
    if lambda_min <= 0:
        raise ValueError("lambda_min must be greater than 0.")
    if lambda_max <= lambda_min:
        raise ValueError("lambda_max must be greater than lambda_min.")
    if grid_size < 100:
        raise ValueError("grid_size must be at least 100.")
    if not output_home_col or not output_away_col:
        raise ValueError("output column names cannot be empty.")

    result = df.copy()

    prob_home = pd.to_numeric(result[prob_home_col], errors="coerce").to_numpy(
        dtype=float
    )
    prob_away = pd.to_numeric(result[prob_away_col], errors="coerce").to_numpy(
        dtype=float
    )
    prob_over25 = pd.to_numeric(result[prob_over25_col], errors="coerce").to_numpy(
        dtype=float
    )

    lambda_grid = np.linspace(lambda_min, lambda_max, int(grid_size), dtype=float)
    prob_over25_grid = 1.0 - poisson.cdf(2, lambda_grid)
    prob_over25_clipped = np.clip(prob_over25, 1e-6, 1.0 - 1e-6)
    total_lambda = (
        np.interp(prob_over25_clipped, prob_over25_grid, lambda_grid) * bias_correction
    )

    denominator = prob_home + prob_away
    share_home = np.divide(
        prob_home,
        denominator,
        out=np.full_like(prob_home, np.nan, dtype=float),
        where=denominator > 0,
    )

    lambda_home = total_lambda * share_home
    lambda_away = total_lambda * (1.0 - share_home)

    invalid = (
        ~np.isfinite(lambda_home)
        | ~np.isfinite(lambda_away)
        | (prob_home < 0)
        | (prob_away < 0)
    )
    if np.any(invalid):
        if errors == "raise":
            bad_rows = result.index[invalid][:5].tolist()
            raise ValueError(
                "Failed to compute baseline lambdas for rows "
                f"{bad_rows}. Ensure valid probabilities and prob_home + prob_away > 0."
            )
        lambda_home[invalid] = np.nan
        lambda_away[invalid] = np.nan

    result[output_home_col] = lambda_home
    result[output_away_col] = lambda_away
    return result
