"""Probability matrix builders and rho calibration for Dixon-Coles models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from matplotlib.axes import Axes

import numpy as np
import pandas as pd
from scipy.stats import poisson


@runtime_checkable
class ProbabilityMatrixBuilder(Protocol):
    """Protocol for building scoreline probability matrices."""

    def build_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """Build and return a normalized scoreline probability matrix."""


class PoissonMatrixBuilder(ProbabilityMatrixBuilder):
    """Build Poisson scoreline matrix with Dixon-Coles low-score correction.

    The implementation is vectorized for backtesting workloads:
    - one goals vector is created with ``np.arange``,
    - home and away PMF vectors are evaluated once each,
    - the base matrix is computed with ``np.outer``,
    - Dixon-Coles correction is applied only to the four low-score cells.
    """

    def __init__(self, *, rho: float, max_goals_matrix: int) -> None:
        """Initialize Poisson matrix builder configuration."""
        rho_value = float(rho)
        max_goals_value = int(max_goals_matrix)
        if not np.isfinite(rho_value):
            raise ValueError("rho must be finite.")
        if max_goals_value < 2:
            raise ValueError("max_goals_matrix must be at least 2.")
        self.rho = rho_value
        self.max_goals_matrix = max_goals_value

    def build_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        """Build normalized scoreline probability matrix.

        Parameters
        ----------
        lambda_home:
            Expected goals for the home team.
        lambda_away:
            Expected goals for the away team.

        Returns
        -------
        np.ndarray
            A 2D matrix of shape ``(max_goals_matrix + 1, max_goals_matrix + 1)``
            where ``matrix[i, j]`` is the probability of score ``i:j``.

        Notes
        -----
        Dixon-Coles correction is only needed for outcomes:
        ``(0,0)``, ``(0,1)``, ``(1,0)``, ``(1,1)``.
        """
        if not np.isfinite(lambda_home) or not np.isfinite(lambda_away):
            raise ValueError("lambda_home and lambda_away must be finite.")
        if lambda_home < 0 or lambda_away < 0:
            raise ValueError("lambda_home and lambda_away must be non-negative.")
        size = self.max_goals_matrix + 1
        goals = np.arange(size)
        p_home = poisson.pmf(goals, lambda_home)
        p_away = poisson.pmf(goals, lambda_away)
        # Independent Poisson assumptions: P(X=i, Y=j) = P(X=i) * P(Y=j).
        matrix = np.outer(p_home, p_away)

        # Dixon-Coles low-score correction affects only these four outcomes.
        matrix[0, 0] *= 1.0 - (lambda_home * lambda_away * self.rho)
        matrix[0, 1] *= 1.0 + (lambda_home * self.rho)
        matrix[1, 0] *= 1.0 + (lambda_away * self.rho)
        matrix[1, 1] *= 1.0 - self.rho

        matrix = np.clip(matrix, 0.0, None)
        matrix_sum = float(matrix.sum())
        if matrix_sum <= 0:
            raise ValueError("probability matrix sum is not positive.")
        return matrix / matrix_sum


# ---------------------------------------------------------------------------
# Rho calibration
# ---------------------------------------------------------------------------


def average_scoreline_nll(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    *,
    rho: float,
    max_goals_matrix: int,
) -> tuple[float, int]:
    """Average negative log-likelihood of observed scorelines under Dixon-Coles.

    Uses :class:`PoissonMatrixBuilder` with the given ``rho`` to obtain the
    probability of the realized ``(home, away)`` score in each match.
    Only rows with observed goals in ``[0, max_goals_matrix]`` participate.
    Rows where the matrix cannot be built (invalid lambdas) are skipped; the
    returned average divides by the count of successful evaluations.

    Parameters
    ----------
    lambda_home, lambda_away:
        Expected goals per match (same length).
    actual_home, actual_away:
        Observed goals per match.
    rho:
        Dixon-Coles low-score correction parameter.
    max_goals_matrix:
        Inclusive cap on goals per team in the probability matrix.

    Returns
    -------
    tuple[float, int]
        ``(mean_nll, n_used)`` where ``n_used`` is the number of matches with a
        valid matrix and log-probability.
    """
    lam_h = np.asarray(lambda_home, dtype=np.float64)
    lam_a = np.asarray(lambda_away, dtype=np.float64)
    real_h = np.asarray(actual_home, dtype=np.intp)
    real_a = np.asarray(actual_away, dtype=np.intp)

    if not (lam_h.shape == lam_a.shape == real_h.shape == real_a.shape):
        raise ValueError("All input arrays must have the same shape.")

    mask = (real_h >= 0) & (real_h <= max_goals_matrix) & (real_a >= 0) & (real_a <= max_goals_matrix)
    lam_h = lam_h[mask]
    lam_a = lam_a[mask]
    real_h = real_h[mask]
    real_a = real_a[mask]
    n_masked = int(mask.sum())

    if n_masked == 0:
        raise ValueError("No valid matches after filtering by max_goals_matrix.")

    builder = PoissonMatrixBuilder(rho=float(rho), max_goals_matrix=max_goals_matrix)
    nll_sum = 0.0
    n_used = 0
    for i in range(n_masked):
        try:
            matrix = builder.build_matrix(float(lam_h[i]), float(lam_a[i]))
        except ValueError:
            continue
        prob = max(float(matrix[real_h[i], real_a[i]]), 1e-15)
        nll_sum += -np.log(prob)
        n_used += 1

    if n_used == 0:
        raise ValueError("No match produced a valid probability matrix for NLL.")
    return (nll_sum / n_used, n_used)


def average_points_weighted_scoreline_nll(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    *,
    rho: float,
    max_goals_matrix: int,
    exact_weight: float = 1.0,
    goal_diff_weight: float = 2.0 / 3.0,
    outcome_weight: float = 1.0 / 3.0,
) -> tuple[float, int]:
    """Average weighted scoreline NLL aligned with 3/2/1 score utility.

    For each match, the method builds a Dixon-Coles probability matrix and
    computes weighted probability mass relative to the realized score:
    - exact score receives ``exact_weight``,
    - same goal difference (excluding exact) receives ``goal_diff_weight``,
    - same 1x2 outcome (excluding exact/goal-difference) receives ``outcome_weight``.

    The per-match loss is ``-log(max(weighted_prob, 1e-15))`` and the function
    returns the average across all valid matches.
    """
    lam_h = np.asarray(lambda_home, dtype=np.float64)
    lam_a = np.asarray(lambda_away, dtype=np.float64)
    real_h = np.asarray(actual_home, dtype=np.intp)
    real_a = np.asarray(actual_away, dtype=np.intp)

    if not (lam_h.shape == lam_a.shape == real_h.shape == real_a.shape):
        raise ValueError("All input arrays must have the same shape.")
    if exact_weight <= 0:
        raise ValueError("exact_weight must be greater than 0.")
    if goal_diff_weight < 0 or outcome_weight < 0:
        raise ValueError("goal_diff_weight and outcome_weight must be non-negative.")

    mask = (real_h >= 0) & (real_h <= max_goals_matrix) & (real_a >= 0) & (real_a <= max_goals_matrix)
    lam_h = lam_h[mask]
    lam_a = lam_a[mask]
    real_h = real_h[mask]
    real_a = real_a[mask]
    n_masked = int(mask.sum())

    if n_masked == 0:
        raise ValueError("No valid matches after filtering by max_goals_matrix.")

    builder = PoissonMatrixBuilder(rho=float(rho), max_goals_matrix=max_goals_matrix)
    matrix_size = max_goals_matrix + 1
    goals = np.arange(matrix_size, dtype=np.intp)
    goal_diff_grid = goals[:, None] - goals[None, :]
    outcome_grid = np.sign(goal_diff_grid)

    nll_sum = 0.0
    n_used = 0
    for i in range(n_masked):
        try:
            matrix = builder.build_matrix(float(lam_h[i]), float(lam_a[i]))
        except ValueError:
            continue

        exact_mask = np.zeros_like(matrix, dtype=bool)
        exact_mask[real_h[i], real_a[i]] = True

        actual_diff = int(real_h[i] - real_a[i])
        goal_diff_mask = (goal_diff_grid == actual_diff) & ~exact_mask

        actual_outcome = int(np.sign(actual_diff))
        outcome_mask = (outcome_grid == actual_outcome) & ~exact_mask & ~goal_diff_mask

        weighted_prob = (
            exact_weight * float(matrix[exact_mask].sum())
            + goal_diff_weight * float(matrix[goal_diff_mask].sum())
            + outcome_weight * float(matrix[outcome_mask].sum())
        )
        nll_sum += -np.log(max(weighted_prob, 1e-15))
        n_used += 1

    if n_used == 0:
        raise ValueError("No match produced a valid probability matrix for NLL.")
    return (nll_sum / n_used, n_used)


@dataclass(frozen=True)
class RhoCalibrationResult:
    """Container returned by :func:`calibrate_rho`."""

    best_rho: float
    best_nll: float
    n_matches: int
    grid_df: pd.DataFrame


def calibrate_rho(
    lambda_home: np.ndarray,
    lambda_away: np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    *,
    rho_range: tuple[float, float] = (-0.30, 0.30),
    rho_step: float = 0.01,
    max_goals_matrix: int = 10,
) -> RhoCalibrationResult:
    """Grid-search optimal Dixon-Coles rho via negative log-likelihood.

    For each candidate rho a :class:`PoissonMatrixBuilder` is constructed and
    the average NLL of the true scoreline is computed across all matches whose
    actual goals fall within ``[0, max_goals_matrix]``.

    Parameters
    ----------
    lambda_home:
        Predicted expected goals for the home team (one per match).
    lambda_away:
        Predicted expected goals for the away team (one per match).
    actual_home:
        Observed home goals (one per match).
    actual_away:
        Observed away goals (one per match).
    rho_range:
        ``(lo, hi)`` bounds for the rho grid (both inclusive).
    rho_step:
        Step size between consecutive rho candidates.
    max_goals_matrix:
        Maximum goals per team in the probability matrix.

    Returns
    -------
    RhoCalibrationResult
        Best rho, its NLL, match count, and full grid DataFrame.
    """
    lam_h = np.asarray(lambda_home, dtype=np.float64)
    lam_a = np.asarray(lambda_away, dtype=np.float64)
    real_h = np.asarray(actual_home, dtype=np.intp)
    real_a = np.asarray(actual_away, dtype=np.intp)

    if not (lam_h.shape == lam_a.shape == real_h.shape == real_a.shape):
        raise ValueError("All input arrays must have the same shape.")

    mask = (real_h >= 0) & (real_h <= max_goals_matrix) & \
           (real_a >= 0) & (real_a <= max_goals_matrix)
    lam_h = lam_h[mask]
    lam_a = lam_a[mask]
    real_h = real_h[mask]
    real_a = real_a[mask]
    n_valid = int(mask.sum())

    if n_valid == 0:
        raise ValueError("No valid matches after filtering by max_goals_matrix.")

    rho_grid = np.arange(rho_range[0], rho_range[1] + rho_step * 0.5, rho_step)
    rows: list[dict[str, float]] = []
    nll_counts: list[int] = []

    for rho_val in rho_grid:
        mean_nll, n_used = average_scoreline_nll(
            lam_h, lam_a, real_h, real_a,
            rho=float(rho_val), max_goals_matrix=max_goals_matrix,
        )
        rows.append({"rho": round(float(rho_val), 4), "avg_nll": float(mean_nll)})
        nll_counts.append(n_used)

    grid_df = pd.DataFrame(rows)
    best_idx = int(grid_df["avg_nll"].idxmin())
    n_matches = int(nll_counts[best_idx]) if nll_counts else n_valid
    return RhoCalibrationResult(
        best_rho=float(grid_df.loc[best_idx, "rho"]),
        best_nll=float(grid_df.loc[best_idx, "avg_nll"]),
        n_matches=n_matches,
        grid_df=grid_df,
    )


def plot_rho_calibration(
    result: RhoCalibrationResult,
    *,
    ax: Axes | None = None,
) -> Axes:
    """Plot rho grid-search curve with best-point annotation.

    Parameters
    ----------
    result:
        Output of :func:`calibrate_rho`.
    ax:
        Matplotlib axes to draw on.  When ``None`` a new figure is created.

    Returns
    -------
    plt.Axes
        The axes with the plot (caller can further customize title, etc.).
    """
    import matplotlib.pyplot as plt

    df = result.grid_df
    nll_min = df["avg_nll"].min()
    nll_max = df["avg_nll"].max()
    nll_range = nll_max - nll_min
    pad = nll_range * 0.15

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))

    ax.plot(df["rho"], df["avg_nll"], color="C0", linewidth=1.5, label="Avg NLL")
    ax.axvline(
        x=result.best_rho, color="C3", linestyle="--", linewidth=0.8,
        alpha=0.6, label=f"best \u03c1 = {result.best_rho:.2f}",
    )
    ax.scatter([result.best_rho], [result.best_nll], s=40, color="C3", zorder=5)
    ax.set_ylim(nll_min - pad, nll_max + pad)

    ax.annotate(
        f"\u03c1 = {result.best_rho:.2f}, NLL = {result.best_nll:.5f}",
        xy=(result.best_rho, result.best_nll),
        xytext=(result.best_rho + 0.01, result.best_nll + nll_range * 0.30),
        fontsize=8, color="C3",
    )

    ax.set_xlabel("\u03c1 (rho)")
    ax.set_ylabel("Average Negative Log-Likelihood")
    ax.set_title(
        f"Dixon-Coles \u03c1 Grid Search (N={result.n_matches})",
    )
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.figure.tight_layout()
    return ax
