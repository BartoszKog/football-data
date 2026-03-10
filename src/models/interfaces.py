"""Common interfaces for predictive models in the project.

Public API:
- PredictiveModel
- TrainablePredictiveModel
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd


@runtime_checkable
class PredictiveModel(Protocol):
    """Interface for models that can generate predictions from a DataFrame."""

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return predictions as a DataFrame (or input with added columns)."""


@runtime_checkable
class TrainablePredictiveModel(PredictiveModel, Protocol):
    """Interface for models that require fitting before prediction.

    Two validation styles are supported by the tuning helpers:

    **2-way (row-based walk-forward):** ``run_trainable_grid_search`` calls
    ``fit(train_df)`` — i.e. ``fit(train_df, eval_df=None)`` — then
    ``predict(valid_df)`` on the validation slice of the fold; metrics are
    computed on that validation slice only.

    **3-way (season-based walk-forward):** ``run_trainable_grid_search_three_way``
    calls ``fit(train_df, eval_df=val_df)`` so the model can use the val slice
    only during training (e.g. early stopping), then ``predict(eval_df)`` on a
    separate eval slice; val must not leak into final fold metrics.

    Implementations with legacy signature ``fit(self, df)`` still work at
    runtime with ``isinstance(..., TrainablePredictiveModel)`` because the
    first positional argument maps to ``train_df``. New code should implement
    ``fit(self, train_df, eval_df=None)`` explicitly.

    Implementations should avoid mutating input DataFrames and should keep all
    learned state inside the model instance.
    """

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: pd.DataFrame | None = None,
    ) -> "TrainablePredictiveModel":
        """Fit model on one training fold and return self for chaining.

        Parameters
        ----------
        train_df:
            DataFrame used to estimate model parameters for a single training
            fold.
        eval_df:
            Optional DataFrame used only during training (for example, as an
            early-stopping or calibration set). This dataset should not be
            used for final model evaluation in grid search – out-of-sample
            metrics are computed on a separate evaluation fold instead.
        """
