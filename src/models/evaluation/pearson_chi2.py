"""Pearson chi-square diagnostics for scoreline probability matrices."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from scipy.stats import chi2

from ..components import ProbabilityMatrixBuilder

_BUCKET_LABELS: tuple[str, ...] = ("0", "1", "2", "3+")
_BASE_BIN_LABELS: tuple[str, ...] = tuple(
    f"h{home_bucket}_a{away_bucket}"
    for home_bucket in _BUCKET_LABELS
    for away_bucket in _BUCKET_LABELS
)


@dataclass(frozen=True)
class PearsonChi2ScorelineResult:
    """Container returned by :func:`pearson_chi2_scoreline_gof`.

    Attributes
    ----------
    chi2_stat:
        Pearson chi-square statistic computed on bins used in the final test
        (after optional merge into ``Other``).
    pvalue:
        Right-tail p-value from ``scipy.stats.chi2.sf(chi2_stat, dof)``.
        When ``dof <= 0`` or statistic is non-finite, this is ``NaN``.
    dof:
        Degrees of freedom used for p-value computation:
        ``n_bins_after_merge - 1 - ddof``.
    n_matches:
        Number of matches that produced a valid probability matrix and were
        included in observed/expected aggregation.
    min_expected_threshold:
        Threshold used to mark sparse bins for merge into ``Other``.
    ddof:
        User-supplied degrees-of-freedom adjustment (e.g. fitted parameters on
        the same sample).
    n_bins_before_merge:
        Number of base bins before sparse-bin merge (always 16 for the 4x4 grid).
    n_bins_after_merge:
        Number of bins used in the test after merge (16 or less, with optional
        extra ``Other`` row).
    n_bins_merged:
        Count of base bins moved into ``Other`` due to low expected counts.
    bins_df:
        Per-bin diagnostic table with observed/expected counts, Pearson
        contributions, merge markers, and shares.
    """

    chi2_stat: float
    pvalue: float
    dof: int
    n_matches: int
    min_expected_threshold: float
    ddof: int
    n_bins_before_merge: int
    n_bins_after_merge: int
    n_bins_merged: int
    bins_df: pd.DataFrame


def pearson_chi2_scoreline_gof(
    *,
    lambda_home: Sequence[float] | np.ndarray,
    lambda_away: Sequence[float] | np.ndarray,
    actual_home: Sequence[int] | np.ndarray,
    actual_away: Sequence[int] | np.ndarray,
    matrix_builder: ProbabilityMatrixBuilder,
    min_expected_threshold: float = 5.0,
    ddof: int = 0,
) -> PearsonChi2ScorelineResult:
    """Compute Pearson chi-square scoreline goodness-of-fit diagnostics.

    The expected scoreline distribution is aggregated into a fixed 4x4 grid:
    home and away goals are bucketed as ``0``, ``1``, ``2``, ``3+``. This yields
    16 base bins named ``h{home_bucket}_a{away_bucket}`` (e.g. ``h0_a0``,
    ``h2_a3+``). Sparse bins with expected count below
    ``min_expected_threshold`` are merged into a single ``Other`` bin before the
    final test statistic and p-value are computed.

    Parameters
    ----------
    lambda_home, lambda_away:
        Expected goals per match for home and away teams. Arrays must have the
        same shape and contain finite values accepted by ``matrix_builder``.
    actual_home, actual_away:
        Observed home and away goals per match. Values are rounded to nearest
        integer for bin assignment and then clipped by bucket definition
        (``3+`` tail).
    matrix_builder:
        Probability matrix builder used to obtain per-match scoreline
        distributions. Returned matrices must be square, finite, and support at
        least goals ``0..3`` on both axes.
    min_expected_threshold:
        Minimal expected count required for a base bin to remain separate in
        the test. Bins below threshold are merged into ``Other``.
    ddof:
        Degrees-of-freedom adjustment subtracted from ``k - 1`` where ``k`` is
        the number of bins after merge. Useful when evaluating on the same
        sample used to estimate model parameters.

    Returns
    -------
    PearsonChi2ScorelineResult
        Dataclass with scalar test outputs and detailed per-bin diagnostics.

    Raises
    ------
    ValueError
        If inputs have inconsistent shapes, invalid values, unsupported matrix
        dimensions, or when no match yields a valid probability matrix.
    """
    if min_expected_threshold <= 0:
        raise ValueError("min_expected_threshold must be greater than 0.")
    if ddof < 0:
        raise ValueError("ddof must be greater than or equal to 0.")

    lam_h = np.asarray(lambda_home, dtype=np.float64)
    lam_a = np.asarray(lambda_away, dtype=np.float64)
    real_h = np.asarray(actual_home, dtype=np.float64)
    real_a = np.asarray(actual_away, dtype=np.float64)
    if not (lam_h.shape == lam_a.shape == real_h.shape == real_a.shape):
        raise ValueError("All input arrays must have the same shape.")
    if np.any(~np.isfinite(real_h)) or np.any(~np.isfinite(real_a)):
        raise ValueError("actual_home and actual_away must contain finite values.")
    if np.any(real_h < 0) or np.any(real_a < 0):
        raise ValueError("actual_home and actual_away must be non-negative.")

    observed_counts = np.zeros(16, dtype=np.float64)
    expected_counts = np.zeros(16, dtype=np.float64)
    n_used = 0
    expected_sum = 0.0

    for match_idx in range(lam_h.size):
        try:
            matrix = matrix_builder.build_matrix(float(lam_h[match_idx]), float(lam_a[match_idx]))
        except ValueError:
            continue

        matrix_arr = np.asarray(matrix, dtype=np.float64)
        if matrix_arr.ndim != 2 or matrix_arr.shape[0] != matrix_arr.shape[1]:
            raise ValueError("matrix_builder must return a square 2D probability matrix.")
        if matrix_arr.shape[0] < 4:
            raise ValueError("matrix_builder matrix must support at least goals 0..3.")
        if not np.all(np.isfinite(matrix_arr)):
            raise ValueError("matrix_builder returned non-finite matrix values.")
        matrix_arr = np.clip(matrix_arr, 0.0, None)
        matrix_sum = float(matrix_arr.sum())
        if matrix_sum <= 0:
            continue
        matrix_arr = matrix_arr / matrix_sum

        matrix_size = matrix_arr.shape[0]
        home_support = np.minimum(np.arange(matrix_size, dtype=np.intp), 3)
        away_support = np.minimum(np.arange(matrix_size, dtype=np.intp), 3)
        bucket_grid = home_support[:, None] * 4 + away_support[None, :]
        expected_counts += np.bincount(
            bucket_grid.ravel(),
            weights=matrix_arr.ravel(),
            minlength=16,
        )

        observed_home = int(np.rint(real_h[match_idx]))
        observed_away = int(np.rint(real_a[match_idx]))
        observed_bin = min(observed_home, 3) * 4 + min(observed_away, 3)
        observed_counts[observed_bin] += 1.0

        n_used += 1
        expected_sum += float(matrix_arr.sum())

    if n_used == 0:
        raise ValueError("No match produced a valid probability matrix for Pearson chi-square.")
    if not np.isclose(expected_sum, float(n_used), atol=1e-6):
        raise ValueError("Expected counts do not sum to number of valid matches.")

    base_df = pd.DataFrame(
        {
            "bin": list(_BASE_BIN_LABELS),
            "observed": observed_counts.astype(np.float64),
            "expected": expected_counts.astype(np.float64),
            "is_other": False,
        }
    )

    merged_mask = base_df["expected"] < float(min_expected_threshold)
    n_bins_merged = int(merged_mask.sum())
    n_bins_before_merge = int(base_df.shape[0])
    n_bins_after_merge = n_bins_before_merge

    if n_bins_merged > 0:
        other_row = pd.DataFrame(
            [
                {
                    "bin": "Other",
                    "observed": float(base_df.loc[merged_mask, "observed"].sum()),
                    "expected": float(base_df.loc[merged_mask, "expected"].sum()),
                    "is_other": True,
                }
            ]
        )
        test_df = pd.concat([base_df.loc[~merged_mask].copy(), other_row], ignore_index=True)
        n_bins_after_merge = int(test_df.shape[0])
    else:
        test_df = base_df.copy()

    contributions = _pearson_contributions(
        observed=test_df["observed"].to_numpy(dtype=np.float64),
        expected=test_df["expected"].to_numpy(dtype=np.float64),
    )
    test_df["contribution"] = contributions
    chi2_stat = float(np.sum(contributions))

    dof = int(n_bins_after_merge - 1 - int(ddof))
    pvalue = float(chi2.sf(chi2_stat, dof)) if dof > 0 and np.isfinite(chi2_stat) else float("nan")

    bins_df = base_df.copy()
    bins_df["merged_to_other"] = merged_mask.to_numpy(dtype=bool)
    bins_df["used_in_test"] = ~merged_mask.to_numpy(dtype=bool)
    bins_df["contribution"] = np.nan
    bins_df.loc[~merged_mask, "contribution"] = test_df.loc[
        ~test_df["is_other"], "contribution"
    ].to_numpy(dtype=np.float64)

    if n_bins_merged > 0:
        other_display = test_df.loc[test_df["is_other"]].copy()
        other_display["merged_to_other"] = False
        other_display["used_in_test"] = True
        bins_df = pd.concat([bins_df, other_display], ignore_index=True)

    bins_df["observed_share"] = bins_df["observed"] / float(n_used)
    bins_df["expected_share"] = bins_df["expected"] / float(n_used)

    return PearsonChi2ScorelineResult(
        chi2_stat=chi2_stat,
        pvalue=pvalue,
        dof=dof,
        n_matches=n_used,
        min_expected_threshold=float(min_expected_threshold),
        ddof=int(ddof),
        n_bins_before_merge=n_bins_before_merge,
        n_bins_after_merge=n_bins_after_merge,
        n_bins_merged=n_bins_merged,
        bins_df=bins_df,
    )


def _pearson_contributions(*, observed: np.ndarray, expected: np.ndarray) -> np.ndarray:
    """Return per-bin Pearson chi-square contributions.

    For bins with positive expected count, computes ``(O - E)^2 / E``.
    If ``E == 0`` and ``O > 0``, contribution is set to ``np.inf`` to indicate
    an invalid finite chi-square decomposition for that bin.
    """
    if observed.shape != expected.shape:
        raise ValueError("observed and expected must have the same shape.")
    contributions = np.zeros_like(expected, dtype=np.float64)
    positive_mask = expected > 0.0
    contributions[positive_mask] = (
        (observed[positive_mask] - expected[positive_mask]) ** 2
    ) / expected[positive_mask]
    invalid_mask = ~positive_mask & (observed > 0.0)
    contributions[invalid_mask] = np.inf
    return contributions
