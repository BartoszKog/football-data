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

    Expected usage in time-series validation:
    1. call ``fit(train_df)`` on each training fold,
    2. call ``predict(valid_df)`` on the matching validation fold.

    Implementations should avoid mutating input DataFrames and should keep all
    learned state inside the model instance.
    """

    def fit(self, df: pd.DataFrame) -> "TrainablePredictiveModel":
        """Fit model on one training fold and return self for chaining."""
