"""Tests for PIT diagnostics on scoreline probability matrices.

Run with: python -m unittest tests.models.evaluation.test_pit -v
"""

import os
import sys
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from src.models import (
    build_pit_components,
    build_pit_diagnostics,
    get_pit_variant,
    plot_pit_histogram_replicates,
    plot_pit_worm_replicates,
    randomized_pit_replicates_from_components,
)
from src.models.components import PoissonMatrixBuilder


class _FixedMatrixBuilder:
    """Minimal matrix builder for deterministic PIT variant tests."""

    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = np.asarray(matrix, dtype=float)

    def build_matrix(self, lambda_home: float, lambda_away: float) -> np.ndarray:
        del lambda_home, lambda_away
        return self.matrix / self.matrix.sum()


def _fixed_matrix() -> np.ndarray:
    return np.array(
        [
            [0.08, 0.07, 0.03],
            [0.11, 0.16, 0.09],
            [0.06, 0.22, 0.18],
        ],
        dtype=float,
    )


class TestPitVariants(unittest.TestCase):
    def test_builtin_variants_produce_normalized_distributions(self):
        matrix = _fixed_matrix()
        for variant_name in [
            "home_goals",
            "away_goals",
            "away_given_home",
            "home_given_away",
            "total_goals",
            "goal_difference",
        ]:
            variant = get_pit_variant(variant_name)
            distribution = variant.extractor(matrix, 1, 1)
            self.assertIsNotNone(distribution)
            self.assertAlmostEqual(float(distribution.probabilities.sum()), 1.0)

    def test_total_goals_and_goal_difference_supports(self):
        matrix = _fixed_matrix()
        total = get_pit_variant("total_goals").extractor(matrix, 1, 1)
        diff = get_pit_variant("goal_difference").extractor(matrix, 1, 1)

        self.assertEqual(total.support.tolist(), [0, 1, 2, 3, 4])
        self.assertEqual(total.observed_value, 2)
        self.assertEqual(diff.support.tolist(), [-2, -1, 0, 1, 2])
        self.assertEqual(diff.observed_value, 0)

    def test_conditional_variants_normalize_selected_slice(self):
        matrix = _fixed_matrix()
        away_given_home = get_pit_variant("away_given_home").extractor(matrix, 1, 2)
        home_given_away = get_pit_variant("home_given_away").extractor(matrix, 2, 1)

        expected_away = matrix[1, :] / matrix[1, :].sum()
        expected_home = matrix[:, 1] / matrix[:, 1].sum()
        np.testing.assert_allclose(away_given_home.probabilities, expected_away)
        np.testing.assert_allclose(home_given_away.probabilities, expected_home)


class TestPitDiagnostics(unittest.TestCase):
    def test_replicates_are_reproducible_for_same_random_states(self):
        components = build_pit_components(
            lambda_home=[1.0, 1.1],
            lambda_away=[0.9, 1.2],
            actual_home=[1, 2],
            actual_away=[1, 0],
            matrix_builder=_FixedMatrixBuilder(_fixed_matrix()),
            variants=["home_goals", "goal_difference"],
        )

        states = np.array([10, 11, 12])
        first = randomized_pit_replicates_from_components(components, states)
        second = randomized_pit_replicates_from_components(components, states)

        self.assertEqual(set(first), {"home_goals", "goal_difference"})
        for key in first:
            np.testing.assert_allclose(first[key], second[key])

    def test_build_pit_diagnostics_with_poisson_matrix_builder(self):
        result = build_pit_diagnostics(
            lambda_home=np.array([1.2, 1.5]),
            lambda_away=np.array([1.0, 0.9]),
            actual_home=np.array([1, 2]),
            actual_away=np.array([1, 0]),
            matrix_builder=PoissonMatrixBuilder(rho=-0.05, max_goals_matrix=8),
            variants=["home_goals", "away_goals", "total_goals"],
            random_states=np.array([100, 101]),
            model_name="poisson_dc",
            sample_name="tiny",
        )

        self.assertEqual(result.summary.shape[0], 3)
        self.assertEqual(set(result.replicates), {"home_goals", "away_goals", "total_goals"})
        self.assertEqual(result.replicates["home_goals"].shape, (2, 2))
        self.assertIn("ks_pvalue", result.summary.columns)

    def test_pit_plots_return_figures(self):
        result = build_pit_diagnostics(
            lambda_home=np.array([1.2, 1.5]),
            lambda_away=np.array([1.0, 0.9]),
            actual_home=np.array([1, 2]),
            actual_away=np.array([1, 0]),
            matrix_builder=PoissonMatrixBuilder(rho=0.0, max_goals_matrix=6),
            variants=["home_goals", "away_goals"],
            random_states=np.array([100, 101, 102]),
            model_name="poisson_dc",
            sample_name="tiny",
        )

        hist_fig = plot_pit_histogram_replicates(result, figsize=(7.0, 4.0))
        worm_fig = plot_pit_worm_replicates(result, n_simulations=20, figsize=(7.0, 4.0))
        self.assertIsInstance(hist_fig, matplotlib.figure.Figure)
        self.assertIsInstance(worm_fig, matplotlib.figure.Figure)
        np.testing.assert_allclose(hist_fig.get_size_inches(), [7.0, 4.0])
        np.testing.assert_allclose(worm_fig.get_size_inches(), [7.0, 4.0])
        plt.close(hist_fig)
        plt.close(worm_fig)

    def test_pit_plots_support_multiple_models_single_variant(self):
        first = build_pit_diagnostics(
            lambda_home=np.array([1.2, 1.5]),
            lambda_away=np.array([1.0, 0.9]),
            actual_home=np.array([1, 2]),
            actual_away=np.array([1, 0]),
            matrix_builder=PoissonMatrixBuilder(rho=0.0, max_goals_matrix=6),
            variants=["home_goals"],
            random_states=np.array([100, 101, 102]),
            model_name="first",
            sample_name="tiny",
        )
        second = build_pit_diagnostics(
            lambda_home=np.array([1.0, 1.4]),
            lambda_away=np.array([1.1, 1.0]),
            actual_home=np.array([0, 1]),
            actual_away=np.array([1, 1]),
            matrix_builder=PoissonMatrixBuilder(rho=-0.05, max_goals_matrix=6),
            variants=["home_goals"],
            random_states=np.array([100, 101, 102]),
            model_name="second",
            sample_name="tiny",
        )

        results = {"First": first, "Second": second}
        hist_fig = plot_pit_histogram_replicates(results)
        worm_fig = plot_pit_worm_replicates(results, n_simulations=20)
        self.assertIsInstance(hist_fig, matplotlib.figure.Figure)
        self.assertIsInstance(worm_fig, matplotlib.figure.Figure)
        plt.close(hist_fig)
        plt.close(worm_fig)


if __name__ == "__main__":
    unittest.main()
