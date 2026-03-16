"""XGBoost Poisson regression model for football scoreline prediction."""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import xgboost as xgb

from ..components import (
    ExpectedPointsOptimizer,
    ExpectedPointsRule,
    PoissonMatrixBuilder,
    ProbabilityMatrixBuilder,
    ScoreOptimizer,
)
from ..interfaces import TrainablePredictiveModel


class XGBoostPoissonModel(TrainablePredictiveModel):
    """Predict football scorelines using paired XGBoost Poisson regressors.

    Two separate ``xgb.XGBRegressor`` instances (one for home goals, one for
    away goals) produce expected-goal lambdas. These lambdas are fed into a
    probability matrix builder and a score optimizer to select the scoreline
    that maximizes expected points.

    The class follows the Dependency Injection pattern: ready-to-use XGBoost
    model instances are passed in at construction time, while the matrix
    builder and optimizer fall back to sensible defaults when not provided.
    """

    def __init__(
        self,
        *,
        features_home: list[str],
        features_away: list[str],
        model_home: xgb.XGBRegressor,
        model_away: xgb.XGBRegressor,
        rho: float = 0.0,
        max_goals_matrix: int = 6,
        max_goals_prediction: int = 4,
        points_rule: ExpectedPointsRule | None = None,
        matrix_builder: ProbabilityMatrixBuilder | None = None,
        optimizer: ScoreOptimizer | None = None,
        errors: Literal["coerce", "raise"] = "coerce",
        verbose: bool = False,
        target_home_col: str = "home_score",
        target_away_col: str = "away_score",
        pred_home_col: str = "pred_home_goals",
        pred_away_col: str = "pred_away_goals",
    ) -> None:
        """Initialize model with injected XGBoost regressors and components.

        Parameters
        ----------
        features_home:
            Column names used as input features for the home-goals model.
        features_away:
            Column names used as input features for the away-goals model.
        model_home:
            Pre-configured ``xgb.XGBRegressor`` for predicting home goals.
        model_away:
            Pre-configured ``xgb.XGBRegressor`` for predicting away goals.
        rho:
            Dixon-Coles low-score correction parameter, used only when
            constructing the default ``matrix_builder``.
        max_goals_matrix:
            Maximum goals per team in the probability matrix, used only
            when constructing the default ``matrix_builder`` and ``optimizer``.
        max_goals_prediction:
            Maximum goals per team considered while selecting the final
            prediction, used only when constructing the default ``optimizer``.
        points_rule:
            Scoring rule for the default optimizer. Ignored when a custom
            ``optimizer`` is provided.
        matrix_builder:
            Probability matrix builder. When ``None``, a
            ``PoissonMatrixBuilder`` is created from ``rho`` and
            ``max_goals_matrix``.
        optimizer:
            Score optimizer. When ``None``, an ``ExpectedPointsOptimizer``
            is created from ``points_rule``, ``max_goals_prediction`` and
            ``max_goals_matrix``.
        errors:
            Row-level error policy: ``"coerce"`` produces NaN predictions
            for invalid rows, ``"raise"`` raises with row-index context.
        verbose:
            Verbosity flag passed to ``XGBRegressor.fit()``.
        target_home_col:
            Column name for home-team goals in the training DataFrame.
        target_away_col:
            Column name for away-team goals in the training DataFrame.
        pred_home_col:
            Output column name for predicted home goals.
        pred_away_col:
            Output column name for predicted away goals.
        """
        self.features_home = list(features_home)
        self.features_away = list(features_away)
        self.model_home = model_home
        self.model_away = model_away
        self.errors = errors
        self.verbose = verbose
        self.target_home_col = target_home_col
        self.target_away_col = target_away_col
        self.pred_home_col = pred_home_col
        self.pred_away_col = pred_away_col

        if matrix_builder is None:
            matrix_builder = PoissonMatrixBuilder(
                rho=rho,
                max_goals_matrix=max_goals_matrix,
            )
        if optimizer is None:
            optimizer = ExpectedPointsOptimizer(
                rules=points_rule or ExpectedPointsRule(),
                max_goals_prediction=max_goals_prediction,
                max_goals_matrix=max_goals_matrix,
            )
        if not isinstance(matrix_builder, ProbabilityMatrixBuilder):
            raise TypeError(
                "matrix_builder must implement ProbabilityMatrixBuilder."
            )
        if not isinstance(optimizer, ScoreOptimizer):
            raise TypeError("optimizer must implement ScoreOptimizer.")

        self.matrix_builder: ProbabilityMatrixBuilder = matrix_builder
        self.optimizer: ScoreOptimizer = optimizer

    def _validate_required_columns(
        self,
        df: pd.DataFrame,
        required: list[str],
        *,
        where: str,
    ) -> None:
        """Validate that all ``required`` columns are present in ``df``.

        Parameters
        ----------
        df:
            DataFrame to validate.
        required:
            Column names that must be present in ``df``.
        where:
            Human-readable context used in the error message, e.g.
            ``\"train_df in fit()\"`` or ``\"df in predict()\"``.

        Raises
        ------
        KeyError
            If one or more required columns are missing.
        """
        missing = [col for col in required if col not in df.columns]
        if missing:
            cols = ", ".join(missing)
            raise KeyError(
                f"Missing required columns in {where}: {cols}. "
                "Check that the feature engineering pipeline produced these "
                "columns before calling the model."
            )

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> "XGBoostPoissonModel":
        """Fit home and away XGBoost regressors on the training data.

        Parameters
        ----------
        train_df:
            Training DataFrame containing feature columns and target columns.
        eval_df:
            Optional validation DataFrame for early stopping. When provided,
            it is passed as ``eval_set`` to both XGBoost models.

        Returns
        -------
        self
        """
        self._validate_required_columns(
            train_df,
            [
                *self.features_home,
                *self.features_away,
                self.target_home_col,
                self.target_away_col,
            ],
            where="train_df in XGBoostPoissonModel.fit()",
        )

        X_train_home = train_df[self.features_home]
        X_train_away = train_df[self.features_away]
        y_train_home = train_df[self.target_home_col]
        y_train_away = train_df[self.target_away_col]

        fit_kwargs_home: dict[str, object] = {"verbose": self.verbose}
        fit_kwargs_away: dict[str, object] = {"verbose": self.verbose}

        if eval_df is not None:
            self._validate_required_columns(
                eval_df,
                [
                    *self.features_home,
                    *self.features_away,
                    self.target_home_col,
                    self.target_away_col,
                ],
                where="eval_df in XGBoostPoissonModel.fit()",
            )

            X_val_home = eval_df[self.features_home]
            X_val_away = eval_df[self.features_away]
            y_val_home = eval_df[self.target_home_col]
            y_val_away = eval_df[self.target_away_col]

            fit_kwargs_home["eval_set"] = [(X_val_home, y_val_home)]
            fit_kwargs_away["eval_set"] = [(X_val_away, y_val_away)]

        self.model_home.fit(X_train_home, y_train_home, **fit_kwargs_home)
        self.model_away.fit(X_train_away, y_train_away, **fit_kwargs_away)

        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy of DataFrame with model score predictions.

        Added columns (default names, configurable via ``__init__``):

        - ``pred_home_goals``
        - ``pred_away_goals``
        - ``pred_score``
        - ``pred_xpts``
        - ``exp_goals_home``
        - ``exp_goals_away``

        Parameters
        ----------
        df:
            DataFrame containing the feature columns used during training.
        """
        self._validate_required_columns(
            df,
            [*self.features_home, *self.features_away],
            where="df in XGBoostPoissonModel.predict()",
        )

        X_test_home = df[self.features_home]
        X_test_away = df[self.features_away]

        lambdas_home: np.ndarray = self.model_home.predict(X_test_home)
        lambdas_away: np.ndarray = self.model_away.predict(X_test_away)

        output_rows: list[dict[str, object]] = []
        for idx, (l_h, l_a) in enumerate(zip(lambdas_home, lambdas_away)):
            try:
                matrix = self.matrix_builder.build_matrix(
                    lambda_home=float(l_h),
                    lambda_away=float(l_a),
                )
                pred_home, pred_away, pred_xpts = self.optimizer.optimize(matrix)
                output_rows.append(
                    {
                        self.pred_home_col: int(pred_home),
                        self.pred_away_col: int(pred_away),
                        "pred_score": f"{pred_home}:{pred_away}",
                        "pred_xpts": float(pred_xpts),
                        "exp_goals_home": float(l_h),
                        "exp_goals_away": float(l_a),
                    }
                )
            except Exception as exc:
                if self.errors == "raise":
                    row_label = df.index[idx]
                    raise ValueError(
                        f"Failed to predict row {row_label}: {exc}"
                    ) from exc
                output_rows.append(
                    {
                        self.pred_home_col: pd.NA,
                        self.pred_away_col: pd.NA,
                        "pred_score": pd.NA,
                        "pred_xpts": pd.NA,
                        "exp_goals_home": pd.NA,
                        "exp_goals_away": pd.NA,
                    }
                )

        output = pd.DataFrame(output_rows, index=df.index)
        return pd.concat([df.copy(), output], axis=1, verify_integrity=True)
