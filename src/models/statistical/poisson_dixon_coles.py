"""Poisson scoreline model with Dixon-Coles low-score correction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import poisson

from ..interfaces import PredictiveModel


@dataclass(frozen=True)
class _ExpectedPointsRule:
    """Internal point rule used for expected points maximization."""

    exact: int = 3
    goal_diff: int = 2
    outcome: int = 1
    miss: int = 0


class PoissonDixonColesModel(PredictiveModel):
    """Predict football scorelines from probabilities with Dixon-Coles correction.

    The model expects per-match probabilities from the feature layer:
    - home-win probability,
    - away-win probability,
    - over-2.5-goals probability.

    It then:
    1. maps over-2.5 probability to total expected goals,
    2. splits expected goals between teams with home/away win probabilities,
    3. builds a Poisson matrix corrected by Dixon-Coles `rho`,
    4. picks scoreline maximizing expected points under Supertyper-like scoring.
    """

    def __init__(
        self,
        *,
        prob_home_col: str = "prob_1",
        prob_away_col: str = "prob_2",
        prob_over25_col: str = "prob_over_25",
        rho: float = 0.0,
        bias_correction: float = 1.0,
        max_goals_matrix: int = 6,
        max_goals_prediction: int = 4,
        errors: Literal["coerce", "raise"] = "coerce",
    ) -> None:
        """Initialize model configuration.

        Parameters
        ----------
        prob_home_col:
            DataFrame column with probability of home win.
        prob_away_col:
            DataFrame column with probability of away win.
        prob_over25_col:
            DataFrame column with probability of over 2.5 goals.
        rho:
            Dixon-Coles low-score correction parameter.
        bias_correction:
            Multiplicative adjustment applied to total expected goals.
        max_goals_matrix:
            Maximum goals per team in the probability matrix (inclusive).
        max_goals_prediction:
            Maximum goals per team considered while selecting final prediction
            (inclusive). Must be less than or equal to `max_goals_matrix`.
        errors:
            Row-level error policy:
            - ``"coerce"`` returns NaN predictions for invalid rows,
            - ``"raise"`` raises with row index context.
        """
        self.prob_home_col = prob_home_col
        self.prob_away_col = prob_away_col
        self.prob_over25_col = prob_over25_col
        self.rho = float(rho)
        self.bias_correction = float(bias_correction)
        self.max_goals_matrix = int(max_goals_matrix)
        self.max_goals_prediction = int(max_goals_prediction)
        self.errors = errors

        self._validate_configuration()
        self._points_rule = _ExpectedPointsRule()

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of DataFrame with model score predictions.

        Added columns:
        - ``pred_home_goals``
        - ``pred_away_goals``
        - ``pred_score``
        - ``pred_xpts``
        - ``exp_goals_home``
        - ``exp_goals_away``
        """
        self._validate_required_columns(df)

        output_rows: list[dict[str, object]] = []
        for row_idx, row in df.iterrows():
            try:
                result = self._predict_row(
                    prob_home=float(row[self.prob_home_col]),
                    prob_away=float(row[self.prob_away_col]),
                    prob_over25=float(row[self.prob_over25_col]),
                )
            except Exception as exc:
                if self.errors == "raise":
                    raise ValueError(
                        f"Failed to predict row {row_idx}: {exc}"
                    ) from exc
                result = {
                    "pred_home_goals": np.nan,
                    "pred_away_goals": np.nan,
                    "pred_score": pd.NA,
                    "pred_xpts": np.nan,
                    "exp_goals_home": np.nan,
                    "exp_goals_away": np.nan,
                }
            output_rows.append(result)

        output = pd.DataFrame(output_rows, index=df.index)
        return pd.concat([df.copy(), output], axis=1)

    def _validate_configuration(self) -> None:
        if self.errors not in {"coerce", "raise"}:
            raise ValueError("errors must be either 'coerce' or 'raise'.")
        if self.bias_correction <= 0:
            raise ValueError("bias_correction must be greater than 0.")
        if self.max_goals_matrix < 2:
            raise ValueError("max_goals_matrix must be at least 2.")
        if self.max_goals_prediction < 0:
            raise ValueError("max_goals_prediction cannot be negative.")
        if self.max_goals_prediction > self.max_goals_matrix:
            raise ValueError(
                "max_goals_prediction cannot exceed max_goals_matrix."
            )
        if not np.isfinite(self.rho):
            raise ValueError("rho must be finite.")

    def _validate_required_columns(self, df: pd.DataFrame) -> None:
        required = [self.prob_home_col, self.prob_away_col, self.prob_over25_col]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required model input columns: {missing}")

    def _predict_row(
        self,
        *,
        prob_home: float,
        prob_away: float,
        prob_over25: float,
    ) -> dict[str, object]:
        self._validate_input_probabilities(prob_home, prob_away, prob_over25)

        total_lambda = self._solve_total_lambda_from_over25(prob_over25)
        total_lambda *= self.bias_correction

        lambda_home, lambda_away = self._split_total_lambda(
            total_lambda=total_lambda,
            prob_home=prob_home,
            prob_away=prob_away,
        )

        matrix = self._build_probability_matrix(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
        )
        pred_home, pred_away, pred_xpts = self._select_best_score(matrix)

        return {
            "pred_home_goals": int(pred_home),
            "pred_away_goals": int(pred_away),
            "pred_score": f"{pred_home}:{pred_away}",
            "pred_xpts": float(pred_xpts),
            "exp_goals_home": float(lambda_home),
            "exp_goals_away": float(lambda_away),
        }

    @staticmethod
    def _validate_input_probabilities(
        prob_home: float,
        prob_away: float,
        prob_over25: float,
    ) -> None:
        values = (prob_home, prob_away, prob_over25)
        if not all(np.isfinite(v) for v in values):
            raise ValueError("input probabilities must be finite.")
        if prob_home < 0 or prob_away < 0:
            raise ValueError("prob_home and prob_away must be non-negative.")
        if prob_home + prob_away <= 0:
            raise ValueError("prob_home + prob_away must be positive.")
        if prob_over25 <= 0 or prob_over25 >= 1:
            raise ValueError("prob_over25 must be in open interval (0, 1).")

    @staticmethod
    def _solve_total_lambda_from_over25(prob_over25: float) -> float:
        target_prob = float(np.clip(prob_over25, 1e-6, 1 - 1e-6))

        def objective(total_lambda: float) -> float:
            return (1.0 - poisson.cdf(2, total_lambda)) - target_prob

        return float(optimize.brentq(objective, 1e-6, 20.0, maxiter=200))

    @staticmethod
    def _split_total_lambda(
        *,
        total_lambda: float,
        prob_home: float,
        prob_away: float,
    ) -> tuple[float, float]:
        share_home = prob_home / (prob_home + prob_away)
        lambda_home = total_lambda * share_home
        lambda_away = total_lambda * (1.0 - share_home)
        return float(lambda_home), float(lambda_away)

    def _build_probability_matrix(
        self,
        *,
        lambda_home: float,
        lambda_away: float,
    ) -> np.ndarray:
        size = self.max_goals_matrix + 1
        matrix = np.zeros((size, size), dtype=float)

        for home_goals in range(size):
            p_home = poisson.pmf(home_goals, lambda_home)
            for away_goals in range(size):
                p_away = poisson.pmf(away_goals, lambda_away)
                base_prob = p_home * p_away
                correction = self._dixon_coles_tau(
                    home_goals=home_goals,
                    away_goals=away_goals,
                    lambda_home=lambda_home,
                    lambda_away=lambda_away,
                    rho=self.rho,
                )
                matrix[home_goals, away_goals] = base_prob * correction

        matrix = np.clip(matrix, 0.0, None)
        matrix_sum = float(matrix.sum())
        if matrix_sum <= 0:
            raise ValueError("probability matrix sum is not positive.")
        return matrix / matrix_sum

    @staticmethod
    def _dixon_coles_tau(
        *,
        home_goals: int,
        away_goals: int,
        lambda_home: float,
        lambda_away: float,
        rho: float,
    ) -> float:
        if home_goals == 0 and away_goals == 0:
            return 1.0 - (lambda_home * lambda_away * rho)
        if home_goals == 0 and away_goals == 1:
            return 1.0 + (lambda_home * rho)
        if home_goals == 1 and away_goals == 0:
            return 1.0 + (lambda_away * rho)
        if home_goals == 1 and away_goals == 1:
            return 1.0 - rho
        return 1.0

    def _select_best_score(self, matrix: np.ndarray) -> tuple[int, int, float]:
        best_home = 0
        best_away = 0
        best_xpts = -np.inf
        best_point_prob = -np.inf

        for pred_home in range(self.max_goals_prediction + 1):
            for pred_away in range(self.max_goals_prediction + 1):
                expected_points = self._expected_points(
                    pred_home=pred_home,
                    pred_away=pred_away,
                    matrix=matrix,
                )
                direct_prob = float(matrix[pred_home, pred_away])
                if (
                    expected_points > best_xpts
                    or (
                        np.isclose(expected_points, best_xpts)
                        and direct_prob > best_point_prob
                    )
                ):
                    best_xpts = expected_points
                    best_point_prob = direct_prob
                    best_home = pred_home
                    best_away = pred_away

        return best_home, best_away, float(best_xpts)

    def _expected_points(
        self,
        *,
        pred_home: int,
        pred_away: int,
        matrix: np.ndarray,
    ) -> float:
        expected = 0.0
        rows, cols = matrix.shape
        for real_home in range(rows):
            for real_away in range(cols):
                probability = float(matrix[real_home, real_away])
                points = self._score_points(
                    pred_home=pred_home,
                    pred_away=pred_away,
                    real_home=real_home,
                    real_away=real_away,
                )
                expected += probability * points
        return float(expected)

    def _score_points(
        self,
        *,
        pred_home: int,
        pred_away: int,
        real_home: int,
        real_away: int,
    ) -> int:
        if pred_home == real_home and pred_away == real_away:
            return self._points_rule.exact

        pred_diff = pred_home - pred_away
        real_diff = real_home - real_away
        if pred_diff == real_diff:
            return self._points_rule.goal_diff

        pred_outcome = int(np.sign(pred_diff))
        real_outcome = int(np.sign(real_diff))
        if pred_outcome == real_outcome:
            return self._points_rule.outcome
        return self._points_rule.miss
