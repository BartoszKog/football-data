"""Poisson scoreline model with Dixon-Coles low-score correction."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import poisson

from ..components import (
    ExpectedPointsOptimizer,
    ExpectedPointsRule,
    PoissonMatrixBuilder,
    ProbabilityMatrixBuilder,
    ScoreOptimizer,
)
from ..interfaces import PredictiveModel


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
        points_rule: ExpectedPointsRule | None = None,
        matrix_builder: ProbabilityMatrixBuilder | None = None,
        optimizer: ScoreOptimizer | None = None,
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
        points_rule:
            Point rule used by default expected-points optimizer. Ignored when
            a custom ``optimizer`` is provided.
        matrix_builder:
            Optional probability matrix builder dependency. If omitted, the
            model uses ``PoissonMatrixBuilder`` configured with ``rho`` and
            ``max_goals_matrix``.
        optimizer:
            Optional score optimizer dependency. If omitted, the model uses
            ``ExpectedPointsOptimizer`` configured with ``points_rule``,
            ``max_goals_prediction`` and ``max_goals_matrix``.
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
        if matrix_builder is None:
            matrix_builder = PoissonMatrixBuilder(
                rho=self.rho,
                max_goals_matrix=self.max_goals_matrix,
            )
        if optimizer is None:
            optimizer = ExpectedPointsOptimizer(
                rules=points_rule or ExpectedPointsRule(),
                max_goals_prediction=self.max_goals_prediction,
                max_goals_matrix=self.max_goals_matrix,
            )
        if not isinstance(matrix_builder, ProbabilityMatrixBuilder):
            raise TypeError(
                "matrix_builder must implement ProbabilityMatrixBuilder."
            )
        if not isinstance(optimizer, ScoreOptimizer):
            raise TypeError("optimizer must implement ScoreOptimizer.")

        self.matrix_builder: ProbabilityMatrixBuilder = matrix_builder
        self.optimizer: ScoreOptimizer = optimizer

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
        home_col_idx = int(df.columns.get_loc(self.prob_home_col)) + 1
        away_col_idx = int(df.columns.get_loc(self.prob_away_col)) + 1
        over25_col_idx = int(df.columns.get_loc(self.prob_over25_col)) + 1
        for row_values in df.itertuples(index=True, name=None):
            row_idx = row_values[0]
            try:
                result = self._predict_row(
                    prob_home=float(row_values[home_col_idx]),
                    prob_away=float(row_values[away_col_idx]),
                    prob_over25=float(row_values[over25_col_idx]),
                )
            except Exception as exc:
                if self.errors == "raise":
                    raise ValueError(
                        f"Failed to predict row {row_idx}: {exc}"
                    ) from exc
                result = {
                    "pred_home_goals": pd.NA,
                    "pred_away_goals": pd.NA,
                    "pred_score": pd.NA,
                    "pred_xpts": pd.NA,
                    "exp_goals_home": pd.NA,
                    "exp_goals_away": pd.NA,
                }
            output_rows.append(result)

        output = pd.DataFrame(output_rows, index=df.index)
        return pd.concat([df.copy(), output], axis=1)

    def _validate_configuration(self) -> None:
        if self.errors not in {"coerce", "raise"}:
            raise ValueError("errors must be either 'coerce' or 'raise'.")
        if not np.isfinite(self.bias_correction):
            raise ValueError("bias_correction must be finite.")
        if self.bias_correction <= 0:
            raise ValueError("bias_correction must be greater than 0.")

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

        matrix = self.matrix_builder.build_matrix(
            lambda_home=lambda_home,
            lambda_away=lambda_away,
        )
        pred_home, pred_away, pred_xpts = self.optimizer.optimize(matrix)

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

