"""Smoke tests for season walk-forward splits and three-way trainable grid search.

Run with: python -m unittest tests.test_time_splits_and_grid_search -v
"""

import unittest
from typing import Optional

import pandas as pd

import sys
import os

# Dodaj katalog nadrzędny do ścieżki
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '.')))

from src.models.tuning.time_splits import make_season_walk_forward_splits
from src.models.tuning.trainable_grid_search import run_trainable_grid_search_three_way


class _DummyThreeWayModel:
    """Minimal trainable model for grid search plumbing only."""

    def __init__(self, bias: float = 0.0) -> None:
        self.bias = float(bias)

    def fit(
        self,
        train_df: pd.DataFrame,
        eval_df: Optional[pd.DataFrame] = None,
    ) -> "_DummyThreeWayModel":
        self._train_mean = float(train_df["home_score"].mean())
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        pred_value = self._train_mean + self.bias
        result = df.copy()
        result["pred_home_goals"] = pred_value
        result["pred_away_goals"] = pred_value
        return result


def _build_tiny_season_df() -> pd.DataFrame:
    """One row per season S1, S2, S3 with distinct scores."""
    rows = [
        {
            "match_date": pd.Timestamp("2020-01-01"),
            "season": "S1",
            "home_score": 0,
            "away_score": 2,
        },
        {
            "match_date": pd.Timestamp("2021-01-01"),
            "season": "S2",
            "home_score": 1,
            "away_score": 1,
        },
        {
            "match_date": pd.Timestamp("2022-01-01"),
            "season": "S3",
            "home_score": 2,
            "away_score": 0,
        },
    ]
    return pd.DataFrame(rows)


class TestSeasonWalkForwardSplits(unittest.TestCase):
    def test_make_season_walk_forward_splits_raises_when_season_missing_from_df(self):
        # seasons_order includes S1 but df has only S2/S3 — validated before any fold.
        df = pd.DataFrame(
            [
                {"match_date": pd.Timestamp("2021-01-01"), "season": "S2", "home_score": 1, "away_score": 1},
                {"match_date": pd.Timestamp("2022-01-01"), "season": "S3", "home_score": 2, "away_score": 0},
            ]
        )
        with self.assertRaises(ValueError) as ctx:
            make_season_walk_forward_splits(
                df,
                season_col="season",
                seasons_order=["S1", "S2", "S3"],
            )
        self.assertIn("not present", str(ctx.exception).lower())

    def test_make_season_walk_forward_splits_strict_true_ok_when_all_seasons_present(self):
        # strict=True only affects degenerate folds inside the loop; valid data yields same folds.
        df = _build_tiny_season_df()
        folds = make_season_walk_forward_splits(
            df,
            season_col="season",
            seasons_order=["S1", "S2", "S3"],
            strict=True,
        )
        self.assertEqual(len(folds), 1)

    def test_make_season_walk_forward_splits_basic(self):
        df = _build_tiny_season_df()
        folds = make_season_walk_forward_splits(
            df,
            season_col="season",
            seasons_order=["S1", "S2", "S3"],
        )
        self.assertEqual(len(folds), 1)
        fold = folds[0]
        self.assertEqual(list(df["season"].iloc[fold.train_indices]), ["S1"])
        self.assertEqual(list(df["season"].iloc[fold.val_indices]), ["S2"])
        self.assertEqual(list(df["season"].iloc[fold.eval_indices]), ["S3"])


class TestTrainableGridSearchThreeWay(unittest.TestCase):
    def test_run_trainable_grid_search_three_way_smoke(self):
        df = _build_tiny_season_df()
        folds = make_season_walk_forward_splits(
            df,
            season_col="season",
            seasons_order=["S1", "S2", "S3"],
        )
        result = run_trainable_grid_search_three_way(
            model_factory=_DummyThreeWayModel,
            param_grid={"bias": [0.0, 0.5]},
            df=df,
            folds=folds,
            datetime_col="match_date",
            show_progress=False,
        )
        self.assertEqual(result.results_df.shape[0], 2)
        self.assertEqual(set(result.results_df["n_folds"]), {1})


if __name__ == "__main__":
    unittest.main()
