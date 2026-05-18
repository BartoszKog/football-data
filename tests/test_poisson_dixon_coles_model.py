"""Tests for PoissonDixonColesModel bias scaling."""

import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), ".")))

from src.models import PoissonDixonColesModel


class TestPoissonDixonColesBias(unittest.TestCase):
    def test_global_bias_matches_post_split_when_no_per_team_bias(self) -> None:
        """Scaling only bias_correction matches multiplying total before split."""
        prob_home, prob_away, prob_over25 = 0.45, 0.35, 0.55
        c = 1.035
        row_global = PoissonDixonColesModel(
            rho=0.0,
            bias_correction=c,
            use_over25_interpolation=False,
            errors="raise",
        )._predict_row(
            prob_home=prob_home,
            prob_away=prob_away,
            prob_over25=prob_over25,
        )
        row_split = PoissonDixonColesModel(
            rho=0.0,
            bias_correction=c,
            bias_home=None,
            bias_away=None,
            use_over25_interpolation=False,
            errors="raise",
        )._predict_row(
            prob_home=prob_home,
            prob_away=prob_away,
            prob_over25=prob_over25,
        )
        self.assertAlmostEqual(
            float(row_global["exp_goals_home"]),
            float(row_split["exp_goals_home"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(row_global["exp_goals_away"]),
            float(row_split["exp_goals_away"]),
            places=10,
        )

    def test_bias_away_overrides_global_for_away_only(self) -> None:
        prob_home, prob_away, prob_over25 = 0.5, 0.3, 0.52
        m = PoissonDixonColesModel(
            rho=0.0,
            bias_correction=1.02,
            bias_away=0.90,
            use_over25_interpolation=False,
            errors="raise",
        )
        out = m._predict_row(
            prob_home=prob_home,
            prob_away=prob_away,
            prob_over25=prob_over25,
        )
        base = PoissonDixonColesModel(
            rho=0.0,
            bias_correction=1.0,
            use_over25_interpolation=False,
            errors="raise",
        )._predict_row(
            prob_home=prob_home,
            prob_away=prob_away,
            prob_over25=prob_over25,
        )
        self.assertAlmostEqual(
            float(out["exp_goals_home"]),
            1.02 * float(base["exp_goals_home"]),
            places=10,
        )
        self.assertAlmostEqual(
            float(out["exp_goals_away"]),
            0.90 * float(base["exp_goals_away"]),
            places=10,
        )


if __name__ == "__main__":
    unittest.main()
