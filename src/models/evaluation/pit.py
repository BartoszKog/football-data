"""Probability integral transform diagnostics for scoreline distributions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from ..components import ProbabilityMatrixBuilder


@dataclass(frozen=True)
class PITDistribution:
    """One-dimensional discrete distribution used for PIT diagnostics."""

    support: np.ndarray
    probabilities: np.ndarray
    observed_value: int


@dataclass(frozen=True)
class PITVariant:
    """Named transformation from scoreline matrix to a discrete PIT distribution."""

    name: str
    label: str
    extractor: Callable[[np.ndarray, int, int], PITDistribution | None]


@dataclass(frozen=True)
class PITDiagnosticsResult:
    """Container returned by :func:`build_pit_diagnostics`."""

    components: pd.DataFrame
    replicates: dict[str, np.ndarray]
    summary: pd.DataFrame
    variants: tuple[PITVariant, ...]
    random_states: np.ndarray
    model_name: str
    sample_name: str


VariantLike = str | PITVariant


def _goals_support(size: int) -> np.ndarray:
    return np.arange(size, dtype=np.intp)


def _distribution_from_probabilities(
    support: np.ndarray,
    probabilities: np.ndarray,
    observed_value: int,
) -> PITDistribution | None:
    support_arr = np.asarray(support, dtype=np.intp)
    prob_arr = np.asarray(probabilities, dtype=np.float64)
    if support_arr.shape != prob_arr.shape or support_arr.size == 0:
        return None
    if not np.all(np.isfinite(prob_arr)):
        return None
    prob_arr = np.clip(prob_arr, 0.0, None)
    prob_sum = float(prob_arr.sum())
    if prob_sum <= 0:
        return None
    return PITDistribution(
        support=support_arr,
        probabilities=prob_arr / prob_sum,
        observed_value=int(observed_value),
    )


def _extract_home_goals(
    matrix: np.ndarray,
    actual_home: int,
    actual_away: int,
) -> PITDistribution | None:
    del actual_away
    return _distribution_from_probabilities(
        _goals_support(matrix.shape[0]),
        matrix.sum(axis=1),
        actual_home,
    )


def _extract_away_goals(
    matrix: np.ndarray,
    actual_home: int,
    actual_away: int,
) -> PITDistribution | None:
    del actual_home
    return _distribution_from_probabilities(
        _goals_support(matrix.shape[1]),
        matrix.sum(axis=0),
        actual_away,
    )


def _extract_away_given_home(
    matrix: np.ndarray,
    actual_home: int,
    actual_away: int,
) -> PITDistribution | None:
    if actual_home < 0 or actual_home >= matrix.shape[0]:
        return None
    return _distribution_from_probabilities(
        _goals_support(matrix.shape[1]),
        matrix[actual_home, :],
        actual_away,
    )


def _extract_home_given_away(
    matrix: np.ndarray,
    actual_home: int,
    actual_away: int,
) -> PITDistribution | None:
    if actual_away < 0 or actual_away >= matrix.shape[1]:
        return None
    return _distribution_from_probabilities(
        _goals_support(matrix.shape[0]),
        matrix[:, actual_away],
        actual_home,
    )


def _sum_grouped_probabilities(values: np.ndarray, probabilities: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat_values = np.asarray(values, dtype=np.intp).ravel()
    flat_probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
    support = np.unique(flat_values)
    grouped = np.array(
        [flat_probabilities[flat_values == value].sum() for value in support],
        dtype=np.float64,
    )
    return support, grouped


def _extract_total_goals(
    matrix: np.ndarray,
    actual_home: int,
    actual_away: int,
) -> PITDistribution | None:
    home_grid, away_grid = np.indices(matrix.shape)
    support, probabilities = _sum_grouped_probabilities(home_grid + away_grid, matrix)
    return _distribution_from_probabilities(
        support,
        probabilities,
        actual_home + actual_away,
    )


def _extract_goal_difference(
    matrix: np.ndarray,
    actual_home: int,
    actual_away: int,
) -> PITDistribution | None:
    home_grid, away_grid = np.indices(matrix.shape)
    support, probabilities = _sum_grouped_probabilities(home_grid - away_grid, matrix)
    return _distribution_from_probabilities(
        support,
        probabilities,
        actual_home - actual_away,
    )


DEFAULT_PIT_VARIANTS: dict[str, PITVariant] = {
    "home_goals": PITVariant(
        name="home_goals",
        label="Home goals",
        extractor=_extract_home_goals,
    ),
    "away_goals": PITVariant(
        name="away_goals",
        label="Away goals",
        extractor=_extract_away_goals,
    ),
    "away_given_home": PITVariant(
        name="away_given_home",
        label="Away goals | home goals",
        extractor=_extract_away_given_home,
    ),
    "home_given_away": PITVariant(
        name="home_given_away",
        label="Home goals | away goals",
        extractor=_extract_home_given_away,
    ),
    "total_goals": PITVariant(
        name="total_goals",
        label="Total goals",
        extractor=_extract_total_goals,
    ),
    "goal_difference": PITVariant(
        name="goal_difference",
        label="Goal difference",
        extractor=_extract_goal_difference,
    ),
}


def available_pit_variants() -> tuple[str, ...]:
    """Return names of built-in PIT variants."""

    return tuple(DEFAULT_PIT_VARIANTS)


def get_pit_variant(name: str) -> PITVariant:
    """Return a built-in PIT variant by name."""

    try:
        return DEFAULT_PIT_VARIANTS[name]
    except KeyError as exc:
        available = ", ".join(available_pit_variants())
        raise ValueError(f"Unknown PIT variant '{name}'. Available: {available}") from exc


def resolve_pit_variants(variants: Sequence[VariantLike] | None = None) -> tuple[PITVariant, ...]:
    """Resolve string names and custom variants into :class:`PITVariant` objects."""

    selected = ("home_goals", "away_goals", "away_given_home") if variants is None else variants
    resolved: list[PITVariant] = []
    seen: set[str] = set()
    for variant in selected:
        item = get_pit_variant(variant) if isinstance(variant, str) else variant
        if item.name in seen:
            raise ValueError(f"Duplicate PIT variant name: {item.name}")
        seen.add(item.name)
        resolved.append(item)
    if not resolved:
        raise ValueError("At least one PIT variant is required.")
    return tuple(resolved)


def _pit_component_values(distribution: PITDistribution) -> tuple[float, float] | None:
    observed = int(distribution.observed_value)
    support = np.asarray(distribution.support, dtype=np.intp)
    probabilities = np.asarray(distribution.probabilities, dtype=np.float64)
    observed_mask = support == observed
    mass_at_observed = float(probabilities[observed_mask].sum())
    if mass_at_observed <= 0:
        return None
    cdf_before = float(probabilities[support < observed].sum())
    return cdf_before, mass_at_observed


def build_pit_components(
    *,
    lambda_home: Sequence[float] | np.ndarray,
    lambda_away: Sequence[float] | np.ndarray,
    actual_home: Sequence[int] | np.ndarray,
    actual_away: Sequence[int] | np.ndarray,
    matrix_builder: ProbabilityMatrixBuilder,
    variants: Sequence[VariantLike] | None = None,
) -> pd.DataFrame:
    """Build PIT randomization components for each match and variant.

    Each row stores the decomposition needed for randomized PIT:
    ``U = F(y-) + V * P(Y=y)`` where ``V ~ Uniform(0, 1)``.

    Parameters
    ----------
    lambda_home, lambda_away:
        Predicted expected goals per match.
    actual_home, actual_away:
        Observed scoreline per match.
    matrix_builder:
        Scoreline probability matrix builder implementing
        :class:`~src.models.components.ProbabilityMatrixBuilder`.
    variants:
        PIT variants (built-in names or custom :class:`PITVariant` objects).
        When ``None``, defaults from :func:`resolve_pit_variants` are used.

    Returns
    -------
    pd.DataFrame
        One row per usable ``(match, variant)`` with columns including
        ``cdf_before`` and ``mass_at_observed`` for PIT randomization.

    Notes
    -----
    Rows with non-finite inputs, invalid matrices, or unsupported observed
    values for a variant are skipped.
    """

    variant_defs = resolve_pit_variants(variants)
    lam_h = np.asarray(lambda_home, dtype=np.float64)
    lam_a = np.asarray(lambda_away, dtype=np.float64)
    real_h = np.asarray(actual_home, dtype=np.float64)
    real_a = np.asarray(actual_away, dtype=np.float64)

    if not (lam_h.shape == lam_a.shape == real_h.shape == real_a.shape):
        raise ValueError("All input arrays must have the same shape.")

    rows: list[dict[str, object]] = []
    for row_index, (lh, la, raw_ah, raw_aa) in enumerate(
        zip(lam_h, lam_a, real_h, real_a, strict=True)
    ):
        if not (
            np.isfinite(lh)
            and np.isfinite(la)
            and np.isfinite(raw_ah)
            and np.isfinite(raw_aa)
        ):
            continue
        ah = int(round(float(raw_ah)))
        aa = int(round(float(raw_aa)))
        try:
            matrix = matrix_builder.build_matrix(float(lh), float(la))
        except ValueError:
            continue
        for variant in variant_defs:
            distribution = variant.extractor(matrix, int(ah), int(aa))
            if distribution is None:
                continue
            components = _pit_component_values(distribution)
            if components is None:
                continue
            cdf_before, mass_at_observed = components
            rows.append(
                {
                    "match_index": int(row_index),
                    "variant": variant.name,
                    "variant_label": variant.label,
                    "cdf_before": float(cdf_before),
                    "mass_at_observed": float(mass_at_observed),
                    "observed_value": int(distribution.observed_value),
                    "actual_home": int(ah),
                    "actual_away": int(aa),
                    "lambda_home": float(lh),
                    "lambda_away": float(la),
                }
            )

    return pd.DataFrame(
        rows,
        columns=[
            "match_index",
            "variant",
            "variant_label",
            "cdf_before",
            "mass_at_observed",
            "observed_value",
            "actual_home",
            "actual_away",
            "lambda_home",
            "lambda_away",
        ],
    )


def randomized_pit_replicates_from_components(
    components: pd.DataFrame,
    random_states: Sequence[int] | np.ndarray,
    *,
    variants: Sequence[VariantLike] | None = None,
) -> dict[str, np.ndarray]:
    """Generate repeated randomized PIT arrays from PIT components."""

    if components.empty:
        return {variant.name: np.empty((len(random_states), 0)) for variant in resolve_pit_variants(variants)}
    variant_names = (
        [variant.name for variant in resolve_pit_variants(variants)]
        if variants is not None
        else list(dict.fromkeys(components["variant"].astype(str).tolist()))
    )
    states = np.asarray(random_states, dtype=np.int64)
    replicates: dict[str, np.ndarray] = {}
    for variant_name in variant_names:
        subset = components.loc[components["variant"] == variant_name].sort_values("match_index")
        cdf_before = subset["cdf_before"].to_numpy(dtype=np.float64)
        mass_at_observed = subset["mass_at_observed"].to_numpy(dtype=np.float64)
        values = np.empty((states.size, subset.shape[0]), dtype=np.float64)
        for row_idx, random_state in enumerate(states):
            rng = np.random.default_rng(int(random_state))
            values[row_idx, :] = cdf_before + rng.random(subset.shape[0]) * mass_at_observed
        replicates[variant_name] = values
    return replicates


def _pit_ks_uniform(values: np.ndarray) -> tuple[float, float]:
    clean_values = np.asarray(values, dtype=np.float64)
    clean_values = clean_values[np.isfinite(clean_values)]
    if clean_values.size == 0:
        return np.nan, np.nan
    from scipy.stats import kstest

    result = kstest(clean_values, "uniform")
    return float(result.statistic), float(result.pvalue)


def summarize_pit_uniformity(
    replicates: Mapping[str, np.ndarray],
    *,
    variants: Sequence[VariantLike] | None = None,
    model_name: str = "model",
    sample_name: str = "sample",
    replicate_index: int = 0,
) -> pd.DataFrame:
    """Summarize one deterministic PIT replicate against Uniform(0, 1)."""

    variant_defs = resolve_pit_variants(variants)
    variant_lookup = {variant.name: variant for variant in variant_defs}
    rows: list[dict[str, object]] = []
    for variant_name, values in replicates.items():
        replicate_array = np.asarray(values, dtype=np.float64)
        if replicate_array.ndim != 2:
            raise ValueError("Each replicate array must have shape (n_replicates, n_observations).")
        if replicate_array.shape[0] == 0:
            selected = np.array([], dtype=np.float64)
        else:
            selected = replicate_array[min(replicate_index, replicate_array.shape[0] - 1), :]
        ks_statistic, ks_pvalue = _pit_ks_uniform(selected)
        variant = variant_lookup.get(
            variant_name,
            PITVariant(name=variant_name, label=variant_name, extractor=lambda *_: None),
        )
        rows.append(
            {
                "sample": sample_name,
                "model": model_name,
                "pit_series": variant_name,
                "pit_label": variant.label,
                "n": int(selected.size),
                "mean": float(np.nanmean(selected)) if selected.size else np.nan,
                "std": float(np.nanstd(selected, ddof=1)) if selected.size > 1 else np.nan,
                "uniform_mean_delta": (
                    float(np.nanmean(selected) - 0.5) if selected.size else np.nan
                ),
                "uniform_std_delta": (
                    float(np.nanstd(selected, ddof=1) - np.sqrt(1 / 12))
                    if selected.size > 1
                    else np.nan
                ),
                "ks_statistic": ks_statistic,
                "ks_pvalue": ks_pvalue,
            }
        )
    return pd.DataFrame(rows)


def build_pit_diagnostics(
    *,
    lambda_home: Sequence[float] | np.ndarray,
    lambda_away: Sequence[float] | np.ndarray,
    actual_home: Sequence[int] | np.ndarray,
    actual_away: Sequence[int] | np.ndarray,
    matrix_builder: ProbabilityMatrixBuilder,
    variants: Sequence[VariantLike] | None = None,
    random_states: Sequence[int] | np.ndarray | None = None,
    model_name: str = "model",
    sample_name: str = "sample",
) -> PITDiagnosticsResult:
    """Build complete PIT diagnostics artifacts for scoreline models.

    The workflow is:
    1. extract PIT components per ``(match, variant)``,
    2. generate repeated randomized PIT replicates with provided seeds,
    3. summarize one deterministic replicate against ``Uniform(0, 1)``.

    Parameters
    ----------
    lambda_home, lambda_away:
        Predicted expected goals per match.
    actual_home, actual_away:
        Observed scoreline per match.
    matrix_builder:
        Scoreline matrix builder used to obtain the discrete distribution for
        each PIT variant.
    variants:
        PIT variants to evaluate. Defaults to ``resolve_pit_variants(None)``.
    random_states:
        Seeds for randomized PIT replicates. If ``None``, seeds
        ``10000..10099`` are used.
    model_name, sample_name:
        Metadata propagated to summary outputs.

    Returns
    -------
    PITDiagnosticsResult
        Dataclass with raw components, replicate arrays, summary table, and
        metadata.
    """

    variant_defs = resolve_pit_variants(variants)
    states = np.arange(10_000, 10_100, dtype=np.int64) if random_states is None else np.asarray(random_states, dtype=np.int64)
    if states.size == 0:
        raise ValueError("random_states must contain at least one seed.")
    components = build_pit_components(
        lambda_home=lambda_home,
        lambda_away=lambda_away,
        actual_home=actual_home,
        actual_away=actual_away,
        matrix_builder=matrix_builder,
        variants=variant_defs,
    )
    replicates = randomized_pit_replicates_from_components(
        components,
        states,
        variants=variant_defs,
    )
    summary = summarize_pit_uniformity(
        replicates,
        variants=variant_defs,
        model_name=model_name,
        sample_name=sample_name,
    )
    return PITDiagnosticsResult(
        components=components,
        replicates=replicates,
        summary=summary,
        variants=variant_defs,
        random_states=states,
        model_name=model_name,
        sample_name=sample_name,
    )


def _normalize_result_mapping(
    results: PITDiagnosticsResult | Mapping[str, PITDiagnosticsResult],
) -> dict[str, PITDiagnosticsResult]:
    if isinstance(results, PITDiagnosticsResult):
        return {results.model_name: results}
    return dict(results)


def _variant_labels(results: Mapping[str, PITDiagnosticsResult]) -> dict[str, str]:
    labels: dict[str, str] = {}
    for result in results.values():
        for variant in result.variants:
            labels.setdefault(variant.name, variant.label)
    return labels


def _selected_variant_names(
    results: Mapping[str, PITDiagnosticsResult],
    variants: Sequence[str] | None,
) -> list[str]:
    if variants is not None:
        return list(variants)
    names: list[str] = []
    for result in results.values():
        for variant in result.variants:
            if variant.name not in names:
                names.append(variant.name)
    return names


def _pit_density_reference_band(n: int, *, bins: int = 10, confidence_z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    expected_prob = 1 / bins
    bin_width = 1 / bins
    se_density = np.sqrt(n * expected_prob * (1 - expected_prob)) / (n * bin_width)
    return 1 - confidence_z * se_density, 1 + confidence_z * se_density


def _axes_grid(axes, *, n_rows: int, n_cols: int) -> np.ndarray:
    """Return matplotlib axes as a 2D array with requested shape."""

    axes_array = np.asarray(axes, dtype=object)
    if n_rows == 1 and n_cols == 1:
        return axes_array.reshape(1, 1)
    if n_rows == 1:
        return axes_array.reshape(1, n_cols)
    if n_cols == 1:
        return axes_array.reshape(n_rows, 1)
    return axes_array


def _pit_plot_figsize(
    *,
    n_rows: int,
    n_cols: int,
    figsize: tuple[float, float] | None,
) -> tuple[float, float]:
    """Return explicit or default PIT diagnostic figure size."""

    if figsize is not None:
        return figsize
    return 5.2 * n_cols, 3.2 * n_rows + 1.2


def _histogram_replicate_summary(
    replicate_values: np.ndarray,
    *,
    bins: int = 10,
    interval: tuple[float, float] = (0.10, 0.90),
) -> dict[str, np.ndarray]:
    replicate_array = np.asarray(replicate_values, dtype=np.float64)
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_widths = np.diff(bin_edges)
    densities: list[np.ndarray] = []
    for values in replicate_array:
        clean_values = values[np.isfinite(values)]
        if clean_values.size == 0:
            densities.append(np.full(bins, np.nan))
            continue
        counts, _ = np.histogram(clean_values, bins=bin_edges)
        densities.append(counts / (clean_values.size * bin_widths))
    density_array = np.asarray(densities, dtype=np.float64)
    low_q, high_q = interval
    return {
        "bin_edges": bin_edges,
        "bin_centers": (bin_edges[:-1] + bin_edges[1:]) / 2,
        "bin_widths": bin_widths,
        "median_density": np.nanmedian(density_array, axis=0),
        "low_density": np.nanquantile(density_array, low_q, axis=0),
        "high_density": np.nanquantile(density_array, high_q, axis=0),
    }


def plot_pit_histogram_replicates(
    results: PITDiagnosticsResult | Mapping[str, PITDiagnosticsResult],
    *,
    variants: Sequence[str] | None = None,
    title: str = "Repeated randomized PIT histograms",
    bins: int = 10,
    interval: tuple[float, float] = (0.10, 0.90),
    figsize: tuple[float, float] | None = None,
):
    """Plot median PIT histograms with randomization uncertainty bands.

    Parameters
    ----------
    results:
        One diagnostics result or a mapping of model label to diagnostics
        result. Mappings are rendered as subplot rows.
    variants:
        Variant names to plot as subplot columns. If ``None``, all variants
        available in ``results`` are used.
    title:
        Figure title.
    bins:
        Number of PIT bins in ``[0, 1]``.
    interval:
        Quantile interval used for error bars across randomized replicates,
        for example ``(0.10, 0.90)``.
    figsize:
        Optional explicit matplotlib figure size. If omitted, a size scaled to
        rows/columns is used.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one panel per ``(model, variant)``.

    Notes
    -----
    The dashed horizontal line is the uniform density baseline (``1.0``).
    The gray band approximates a 95% sampling band under uniformity for the
    current sample size, while error bars show PIT-randomization variability
    across replicate seeds.
    """

    import matplotlib.pyplot as plt

    result_map = _normalize_result_mapping(results)
    variant_names = _selected_variant_names(result_map, variants)
    labels = _variant_labels(result_map)
    interval_label = f"{int(interval[0] * 100)}-{int(interval[1] * 100)}% PIT randomization"
    fig, axes = plt.subplots(
        len(result_map),
        len(variant_names),
        figsize=_pit_plot_figsize(
            n_rows=len(result_map),
            n_cols=len(variant_names),
            figsize=figsize,
        ),
        sharex=True,
        sharey=True,
    )
    axes = _axes_grid(axes, n_rows=len(result_map), n_cols=len(variant_names))
    for row_idx, (model_label, result) in enumerate(result_map.items()):
        for col_idx, variant_name in enumerate(variant_names):
            ax = axes[row_idx, col_idx]
            replicate_values = np.asarray(result.replicates[variant_name], dtype=np.float64)
            n_replicates, n_observations = replicate_values.shape
            summary = _histogram_replicate_summary(
                replicate_values,
                bins=bins,
                interval=interval,
            )
            band_low, band_high = _pit_density_reference_band(n_observations, bins=bins)
            if np.isfinite(band_low) and np.isfinite(band_high):
                ax.axhspan(
                    band_low,
                    band_high,
                    color="gray",
                    alpha=0.18,
                    zorder=0,
                    label="~95% uniform band" if row_idx == 0 and col_idx == 0 else None,
                )
            median_density = summary["median_density"]
            low_density = summary["low_density"]
            high_density = summary["high_density"]
            yerr = np.vstack([median_density - low_density, high_density - median_density])
            ax.bar(
                summary["bin_centers"],
                median_density,
                width=summary["bin_widths"] * 0.86,
                align="center",
                color=f"C{row_idx}",
                alpha=0.70,
                edgecolor="white",
                linewidth=0.8,
                zorder=2,
                label="Median histogram density" if row_idx == 0 and col_idx == 0 else None,
            )
            ax.errorbar(
                summary["bin_centers"],
                median_density,
                yerr=yerr,
                fmt="none",
                ecolor="black",
                elinewidth=0.9,
                capsize=2.5,
                capthick=0.9,
                zorder=4,
                label=interval_label if row_idx == 0 and col_idx == 0 else None,
            )
            ax.axhline(
                1.0,
                color="black",
                linestyle="--",
                linewidth=1,
                zorder=3,
                label="Uniform density" if row_idx == 0 and col_idx == 0 else None,
            )
            y_max = np.nanmax(np.concatenate([high_density, np.array([band_high, 1.0])]))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, max(1.2, float(y_max) * 1.15))
            ax.set_title(
                f"{labels.get(variant_name, variant_name)}\n"
                f"n={n_observations}, reps={n_replicates}",
                fontsize=10,
            )
            ax.set_xlabel("Randomized PIT" if row_idx == len(result_map) - 1 else "")
            ax.set_ylabel(f"{model_label}\nDensity" if col_idx == 0 else "")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, y=0.99, fontsize=13, fontweight="semibold")
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=min(len(handles), 4),
            frameon=False,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    return fig


def _pit_worm_reference_band(
    n: int,
    *,
    n_simulations: int = 1000,
    alpha: float = 0.05,
    random_state: int = 20260429,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if n <= 0:
        empty = np.array([], dtype=np.float64)
        return empty, empty, empty
    theoretical = (np.arange(1, n + 1) - 0.5) / n
    rng = np.random.default_rng(random_state)
    simulated = np.sort(rng.uniform(size=(n_simulations, n)), axis=1)
    simulated_worms = simulated - theoretical
    lower, upper = np.quantile(simulated_worms, [alpha / 2, 1 - alpha / 2], axis=0)
    return theoretical, lower, upper


def plot_pit_worm_replicates(
    results: PITDiagnosticsResult | Mapping[str, PITDiagnosticsResult],
    *,
    variants: Sequence[str] | None = None,
    title: str = "Repeated randomized PIT worm plot",
    n_simulations: int = 1000,
    random_state: int = 20260429,
    replicate_alpha: float = 0.06,
    figsize: tuple[float, float] | None = None,
):
    """Plot PIT worm diagnostics from repeated randomized PIT replicates.

    Parameters
    ----------
    results:
        One diagnostics result or a mapping of model label to diagnostics
        result. Mappings are rendered as subplot rows.
    variants:
        Variant names to plot as subplot columns. If ``None``, all variants
        available in ``results`` are used.
    title:
        Figure title.
    n_simulations:
        Number of uniform simulations used to compute the worm reference band.
    random_state:
        Base random seed for simulated worm reference bands.
    replicate_alpha:
        Alpha for individual replicate curves.
    figsize:
        Optional explicit matplotlib figure size. If omitted, a size scaled to
        rows/columns is used.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with one worm panel per ``(model, variant)``.

    Notes
    -----
    The y-axis is ``empirical PIT quantile - theoretical uniform quantile``.
    Median curves outside the simulated uniform band indicate systematic
    calibration deviation. As with any one-dimensional PIT view, local
    dependence effects concentrated in a small scoreline region can be subtle
    and may require complementary scoreline-level diagnostics.
    """

    import matplotlib.pyplot as plt

    result_map = _normalize_result_mapping(results)
    variant_names = _selected_variant_names(result_map, variants)
    labels = _variant_labels(result_map)
    fig, axes = plt.subplots(
        len(result_map),
        len(variant_names),
        figsize=_pit_plot_figsize(
            n_rows=len(result_map),
            n_cols=len(variant_names),
            figsize=figsize,
        ),
        sharex=True,
        sharey=False,
    )
    axes = _axes_grid(axes, n_rows=len(result_map), n_cols=len(variant_names))
    for row_idx, (model_label, result) in enumerate(result_map.items()):
        for col_idx, variant_name in enumerate(variant_names):
            ax = axes[row_idx, col_idx]
            replicate_values = np.asarray(result.replicates[variant_name], dtype=np.float64)
            n_replicates, n_observations = replicate_values.shape
            theoretical, band_low, band_high = _pit_worm_reference_band(
                n_observations,
                n_simulations=n_simulations,
                random_state=random_state + row_idx * 10 + col_idx,
            )
            worm_curves = np.sort(replicate_values, axis=1) - theoretical
            median_worm = np.median(worm_curves, axis=0)
            outside_band = (median_worm < band_low) | (median_worm > band_high)
            ax.fill_between(
                theoretical,
                band_low,
                band_high,
                color="gray",
                alpha=0.20,
                zorder=0,
                label="95% simulated Uniform band" if row_idx == 0 and col_idx == 0 else None,
            )
            for worm_curve in worm_curves:
                ax.plot(
                    theoretical,
                    worm_curve,
                    color=f"C{row_idx}",
                    alpha=replicate_alpha,
                    linewidth=0.6,
                    zorder=1,
                )
            ax.plot(
                theoretical,
                median_worm,
                color=f"C{row_idx}",
                linewidth=2.2,
                zorder=3,
                label="Median randomized worm" if row_idx == 0 and col_idx == 0 else None,
            )
            if outside_band.any():
                ax.scatter(
                    theoretical[outside_band],
                    median_worm[outside_band],
                    s=4,
                    alpha=0.95,
                    color="C3",
                    zorder=6,
                    label="Median outside band" if row_idx == 0 and col_idx == 0 else None,
                )
            max_abs = np.nanmax(np.abs(np.concatenate([worm_curves.ravel(), band_low, band_high])))
            y_limit = max(0.03, float(max_abs) * 1.15)
            ax.set_ylim(-y_limit, y_limit)
            ax.axhline(0.0, color="black", linestyle="--", linewidth=1, zorder=2)
            ax.set_xlim(0, 1)
            ax.set_title(
                f"{labels.get(variant_name, variant_name)}\n"
                f"n={n_observations}, reps={n_replicates}, outside={int(outside_band.sum())}",
                fontsize=10,
            )
            ax.set_xlabel("Uniform theoretical quantile" if row_idx == len(result_map) - 1 else "")
            ax.set_ylabel(f"{model_label}\nEmpirical - theoretical PIT" if col_idx == 0 else "")
            ax.grid(alpha=0.25)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(title, y=0.99, fontsize=13, fontweight="semibold")
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.94),
            ncol=min(len(handles), 3),
            frameon=False,
        )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.86))
    return fig
