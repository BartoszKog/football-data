"""Tests for run_predictive_nll_grid_search and average_scoreline_nll.

Run with: python -m unittest tests.test_predictive_nll_grid_search -v
"""

import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from src.models import PoissonDixonColesModel
from src.models.components import average_scoreline_nll
from src.models.tuning.grid_search import plot_nll_grid_search_2d, run_predictive_nll_grid_search


def _tiny_odds_df() -> pd.DataFrame:
    """Minimal frame with implied-prob columns and scores for Poisson DC."""
    return pd.DataFrame(
        [
            {
                "prob_1": 0.45,
                "prob_2": 0.30,
                "prob_over_25": 0.55,
                "home_score": 1,
                "away_score": 1,
            },
            {
                "prob_1": 0.50,
                "prob_2": 0.25,
                "prob_over_25": 0.60,
                "home_score": 2,
                "away_score": 0,
            },
        ]
    )


class TestAverageScorelineNll(unittest.TestCase):
    def test_average_scoreline_nll_matches_single_rho_build(self):
        lam_h = np.array([1.2, 1.5], dtype=np.float64)
        lam_a = np.array([1.0, 1.1], dtype=np.float64)
        h = np.array([1, 2], dtype=np.intp)
        a = np.array([1, 0], dtype=np.intp)
        rho = -0.1
        max_g = 6
        mean_nll, n_used = average_scoreline_nll(
            lam_h, lam_a, h, a, rho=rho, max_goals_matrix=max_g
        )
        self.assertEqual(n_used, 2)
        self.assertTrue(np.isfinite(mean_nll))
        self.assertGreater(mean_nll, 0.0)


class TestRunPredictiveNllGridSearch(unittest.TestCase):
    def test_run_predictive_nll_grid_search_smoke(self):
        df = _tiny_odds_df()

        def _factory(*, rho: float, bias_correction: float) -> PoissonDixonColesModel:
            return PoissonDixonColesModel(
                rho=rho, bias_correction=bias_correction, use_over25_interpolation=True
            )

        result = run_predictive_nll_grid_search(
            model_factory=_factory,
            param_grid={"rho": [-0.1, 0.0], "bias_correction": [1.0, 1.02]},
            df=df,
            show_progress=False,
        )
        self.assertEqual(result.ranking_metric, "avg_nll")
        self.assertEqual(result.results_df.shape[0], 4)
        self.assertIn("objective_metric", result.results_df.columns)
        best = result.results_df["objective_metric"].min()
        self.assertEqual(result.best_metric, best)
        self.assertTrue(result.results_df["objective_metric"].is_monotonic_increasing)
        # Best row is at top (lowest NLL)
        self.assertEqual(float(result.results_df.iloc[0]["objective_metric"]), result.best_metric)


class TestPlotNllGridSearch2d(unittest.TestCase):
    def test_plot_nll_grid_search_2d_smoke(self):
        results_df = pd.DataFrame(
            {
                "rho": [-0.1, -0.1, 0.0, 0.0],
                "bias_correction": [1.0, 1.02, 1.0, 1.02],
                "objective_metric": [1.5, 1.4, 2.0, 10.0],
            }
        )
        _fig, ax = plt.subplots()
        out = plot_nll_grid_search_2d(
            results_df,
            x_param="bias_correction",
            y_param="rho",
            metric_name="objective_metric",
            ax=ax,
        )
        self.assertIs(out, ax)
        plt.close(_fig)

    def test_plot_nll_high_quantile_invalid(self):
        results_df = pd.DataFrame(
            {
                "rho": [-0.1],
                "bias_correction": [1.0],
                "objective_metric": [1.0],
            }
        )
        with self.assertRaises(ValueError) as ctx:
            plot_nll_grid_search_2d(
                results_df,
                x_param="bias_correction",
                y_param="rho",
                high_quantile=1.1,
            )
        self.assertIn("high_quantile", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
