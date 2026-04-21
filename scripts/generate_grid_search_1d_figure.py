"""Generate 1D grid search figure for documentation.

Produces ``outputs/figures/docs/grid_search_1d_rho.png`` by sweeping the
Dixon-Coles ``rho`` parameter on historical seasons (all except ``current``)
using trimmed average market odds as the source of implied probabilities.

Run with::

    uv run python scripts/generate_grid_search_1d_figure.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import matplotlib.pyplot as plt

from src.data import load_and_add_odds_columns_compact
from src.features import add_power_implied_probabilities_standard_markets
from src.models import (
    PoissonDixonColesModel,
    build_param_grid,
    plot_grid_search_1d,
    run_predictive_grid_search,
)


OUTPUT_PATH = Path("outputs/figures/docs/grid_search_1d_rho.png")


def _model_factory(**params) -> PoissonDixonColesModel:
    return PoissonDixonColesModel(**params, use_over25_interpolation=True)


def main() -> None:
    df = load_and_add_odds_columns_compact()
    df = df[df["season"] != "current"].copy()
    df = add_power_implied_probabilities_standard_markets(df, odds_prefix="trimmed_avg")

    param_grid = build_param_grid(
        {"rho": {"start": -0.30, "stop": 0.02, "step": 0.02}}
    )

    search = run_predictive_grid_search(
        model_factory=_model_factory,
        param_grid=param_grid,
        df=df,
        cache_mode="off",
    )

    ax = plot_grid_search_1d(
        search.results_df,
        param_name="rho",
        metric_name="avg_points",
    )
    ax.figure.set_size_inches(10, 5)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ax.figure.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    plt.close(ax.figure)
    print(f"Saved figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
