"""Feature engineering package for reusable football-data transformations.

Public API:
- add_power_implied_probabilities
- add_power_implied_probabilities_standard_markets
- add_baseline_poisson_lambdas
- add_calibrated_poisson_lambdas
"""

from .odds_probabilities import (
    add_power_implied_probabilities,
    add_power_implied_probabilities_standard_markets,
)
from .poisson_priors import add_baseline_poisson_lambdas, add_calibrated_poisson_lambdas

__all__ = [
    "add_power_implied_probabilities",
    "add_power_implied_probabilities_standard_markets",
    "add_baseline_poisson_lambdas",
    "add_calibrated_poisson_lambdas",
]
