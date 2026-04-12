import marimo

__generated_with = "0.23.0"
app = marimo.App(width="medium")

with app.setup:
    import marimo as mo

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import scipy.stats
    import scipy.sparse
    import json
    import os
    import time

    # pygam 0.9.0 uses the removed `.A` sparse-matrix property (scipy >= 1.14)
    if not hasattr(scipy.sparse.csr_matrix, "A"):
        scipy.sparse.csr_matrix.A = property(lambda self: self.toarray())
        scipy.sparse.csc_matrix.A = property(lambda self: self.toarray())
        scipy.sparse.bsr_matrix.A = property(lambda self: self.toarray())
        scipy.sparse.coo_matrix.A = property(lambda self: self.toarray())

    from pygam import PoissonGAM, s, l, te

    from src.data import load_and_add_odds_columns_compact
    from src.features import (
        add_baseline_poisson_lambdas,
        add_power_implied_probabilities_standard_markets,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 1. Wczytanie i Przygotowanie Danych
    """)
    return


@app.cell
def _():
    df_raw = load_and_add_odds_columns_compact(odds_metrics="trimmed_avg")
    df_wide_probs = df_raw.pipe(
        add_power_implied_probabilities_standard_markets,
        odds_prefix="trimmed_avg",
        output_prefix="prob_trimmed_avg",
    )
    df_wide_full = add_baseline_poisson_lambdas(
        df_wide_probs,
        prob_home_col="prob_trimmed_avg_1",
        prob_away_col="prob_trimmed_avg_2",
        prob_over25_col="prob_trimmed_avg_over_25",
        bias_correction=1.0,
    )
    print(
        f"Wide format — {len(df_wide_full)} meczów, "
        f"{len(df_wide_full.columns)} kolumn"
    )
    return df_wide_full, df_wide_probs


@app.function
def make_long_format(
    df: pd.DataFrame,
    home_suffix: str = "_home",
    away_suffix: str = "_away",
) -> pd.DataFrame:
    """Wide (1 row/match) -> Long (2 rows/match).

    Columns ending with *home_suffix* whose counterpart (*away_suffix*)
    also exists are treated as paired features and renamed to
    ``team_<base>`` / ``opponent_<base>``.  All other columns are kept
    as shared match-level features.  An ``is_home`` flag and a
    ``match_id`` (original index) column are added.
    """
    bases = []
    for col in df.columns:
        if col.endswith(home_suffix):
            base = col[: -len(home_suffix)]
            if base + away_suffix in df.columns:
                bases.append(base)

    paired: set[str] = set()
    for b in bases:
        paired.add(b + home_suffix)
        paired.add(b + away_suffix)
    shared_cols = [c for c in df.columns if c not in paired]

    home_data = df[shared_cols].copy()
    for b in bases:
        home_data[f"team_{b}"] = df[b + home_suffix].values
        home_data[f"opponent_{b}"] = df[b + away_suffix].values
    home_data["is_home"] = 1
    home_data["match_id"] = df.index.to_numpy()

    away_data = df[shared_cols].copy()
    for b in bases:
        away_data[f"team_{b}"] = df[b + away_suffix].values
        away_data[f"opponent_{b}"] = df[b + home_suffix].values
    away_data["is_home"] = 0
    away_data["match_id"] = df.index.to_numpy()

    return pd.concat([home_data, away_data], ignore_index=True)


@app.cell
def _(df_wide_full):
    df_long_full = make_long_format(df_wide_full)
    df_long_full["team_score"] = np.where(
        df_long_full["is_home"] == 1,
        df_long_full["home_score"],
        df_long_full["away_score"],
    )
    print(
        f"Long format — {len(df_long_full)} wierszy "
        f"({len(df_long_full) // 2} meczów)"
    )
    return (df_long_full,)


@app.cell
def _(df_long_full):
    mo.ui.dataframe(df_long_full)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 2. Podział Danych
    """)
    return


@app.cell
def _(df_long_full):
    _features = ["team_baseline_lambda", "opponent_baseline_lambda"]
    _target = "team_score"
    _seasons_train = ["2020/2021", "2021/2022", "2022/2023", "2023/2024"]
    _season_val = "2024/2025"

    _df = df_long_full.dropna(subset=_features + [_target]).copy()
    _train_m = _df["season"].isin(_seasons_train)
    _val_m = _df["season"] == _season_val

    bundle = {
        "features": _features,
        "X_train": _df.loc[_train_m, _features],
        "y_train": _df.loc[_train_m, _target],
        "X_val": _df.loc[_val_m, _features],
        "y_val": _df.loc[_val_m, _target],
        "val_long_meta": _df.loc[
            _val_m, ["match_id", "is_home", _target]
        ].rename(columns={_target: "score"}),
    }

    mo.md(
        f"**Trening:** {len(bundle['X_train'])} wierszy | "
        f"**Walidacja:** {len(bundle['X_val'])} wierszy"
    )
    return (bundle,)


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 3. Panel Sterowania — Kreator Formuły GAM
    """)
    return


@app.cell
def _():
    team_term_type = mo.ui.dropdown(
        options=["Splajn (s)", "Liniowa (l)", "Brak"],
        value="Splajn (s)",
        label="Typ członu — Team Lambda",
    )
    team_n_splines = mo.ui.number(
        start=3, stop=50, step=1, value=20, label="n_splines (Team)",
    )
    team_monotonic = mo.ui.checkbox(
        label="Wymuś rosnącą (monotonic_inc) dla Team Lambda",
    )
    opp_term_type = mo.ui.dropdown(
        options=["Splajn (s)", "Liniowa (l)", "Brak"],
        value="Splajn (s)",
        label="Typ członu — Opponent Lambda",
    )
    opp_n_splines = mo.ui.number(
        start=3, stop=50, step=1, value=20, label="n_splines (Opponent)",
    )
    opp_monotonic = mo.ui.checkbox(
        label="Wymuś malejącą (monotonic_dec) dla Opponent Lambda",
    )
    use_interaction = mo.ui.checkbox(
        label="Dodaj interakcję te(team, opponent)",
    )
    te_n_splines = mo.ui.number(
        start=3, stop=20, step=1, value=5, label="n_splines (te)",
    )
    lam_space = mo.ui.dropdown(
        options=[
            "Szeroka siatka (1e-3 do 1e3)",
            "Wysokie wygładzanie (1e1 do 1e4)",
            "Niskie wygładzanie (1e-4 do 1e0)",
            "Bardzo wysokie wygładzanie (1e5 do 1e8)",
        ],
        value="Szeroka siatka (1e-3 do 1e3)",
        label="Przestrzeń kary (Grid Search λ)",
    )
    lam_n_points = mo.ui.slider(
        start=5, stop=51, step=2, value=11, label="Liczba punktów siatki λ",
    )
    return (
        lam_n_points,
        lam_space,
        opp_monotonic,
        opp_n_splines,
        opp_term_type,
        te_n_splines,
        team_monotonic,
        team_n_splines,
        team_term_type,
        use_interaction,
    )


@app.cell
def _(
    lam_n_points,
    lam_space,
    opp_monotonic,
    opp_n_splines,
    opp_term_type,
    te_n_splines,
    team_monotonic,
    team_n_splines,
    team_term_type,
    use_interaction,
):
    mo.vstack([
        mo.md("**Konfiguracja członów GAM**"),
        mo.hstack([team_term_type, team_n_splines, team_monotonic], align="center"),
        mo.hstack([opp_term_type, opp_n_splines, opp_monotonic], align="center"),
        mo.md("---"),
        mo.hstack([use_interaction, te_n_splines], align="center"),
        mo.md("---"),
        mo.hstack([lam_space, lam_n_points], align="center"),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 4. Budowa Formuły i Trening GAM
    """)
    return


@app.cell
def _(
    lam_n_points,
    lam_space,
    opp_monotonic,
    opp_n_splines,
    opp_term_type,
    te_n_splines,
    team_monotonic,
    team_n_splines,
    team_term_type,
    use_interaction,
):
    _terms = []
    term_info = []

    _team_active = team_term_type.value != "Brak"
    _opp_active = opp_term_type.value != "Brak"

    if _team_active:
        if team_term_type.value == "Splajn (s)":
            _kw = {"n_splines": team_n_splines.value}
            if team_monotonic.value:
                _kw["constraints"] = "monotonic_inc"
            _terms.append(s(0, **_kw))
        else:
            _terms.append(l(0))
        term_info.append({"name": "team_baseline_lambda", "features": [0]})

    if _opp_active:
        if opp_term_type.value == "Splajn (s)":
            _kw = {"n_splines": opp_n_splines.value}
            if opp_monotonic.value:
                _kw["constraints"] = "monotonic_dec"
            _terms.append(s(1, **_kw))
        else:
            _terms.append(l(1))
        term_info.append({"name": "opponent_baseline_lambda", "features": [1]})

    if use_interaction.value:
        _terms.append(te(0, 1, n_splines=te_n_splines.value))
        term_info.append({"name": "te(team, opponent)", "features": [0, 1]})

    mo.stop(
        not _terms,
        mo.md('**Wybierz co najmniej jeden czlon formuly (oba sa "Brak").**'),
    )

    formula = _terms[0]
    for _t in _terms[1:]:
        formula = formula + _t

    _n = lam_n_points.value
    _lam_ranges = {
        "Szeroka siatka (1e-3 do 1e3)": (-3, 3),
        "Wysokie wygładzanie (1e1 do 1e4)": (1, 4),
        "Niskie wygładzanie (1e-4 do 1e0)": (-4, 0),
        "Bardzo wysokie wygładzanie (1e5 do 1e8)": (5, 8),
    }
    _lo, _hi = _lam_ranges[lam_space.value]
    lam_grid = np.logspace(_lo, _hi, _n)
    formula_str = str(formula)

    mo.md(
        f"**Formuła:** `{formula_str}`\n\n"
        f"**Siatka λ:** `{lam_space.value}` — **{_n}** punktów"
    )
    return formula, formula_str, lam_grid, term_info


@app.cell
def _(bundle, formula, lam_grid):
    _X_train = bundle["X_train"].to_numpy()
    _y_train = bundle["y_train"].to_numpy()

    gam_model = PoissonGAM(formula)

    _t0 = time.perf_counter()
    gam_model.gridsearch(_X_train, _y_train, lam=lam_grid)
    training_time = round(time.perf_counter() - _t0, 2)

    mo.md(
        f"**Trening zakończony w {training_time} s.** "
        f"Wybrany λ = `{gam_model.lam}`"
    )
    return gam_model, training_time


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 5. Diagnostyka — „Glass Box"
    """)
    return


@app.cell
def _(gam_model, term_info):
    _edof_rows = []
    _info_idx = 0
    for _i, _term in enumerate(gam_model.terms):
        if _term.isintercept:
            continue
        _label = (
            term_info[_info_idx]["name"]
            if _info_idx < len(term_info)
            else f"term_{_i}"
        )
        _n_coefs = _term.n_coefs
        _edof_slice = gam_model.statistics_["edof_per_coef"]
        _start = sum(t.n_coefs for t in gam_model.terms[:_i])
        _end = _start + _n_coefs
        _term_edof = float(np.sum(_edof_slice[_start:_end]))
        _edof_rows.append({
            "Człon": _label,
            "Liczba współczynników": _n_coefs,
            "EDF (suma)": round(_term_edof, 3),
        })
        _info_idx += 1

    _edof_df = pd.DataFrame(_edof_rows)
    mo.vstack([
        mo.md("## Efektywne Stopnie Swobody (EDF)"),
        mo.ui.table(_edof_df, selection=None),
    ])
    return


@app.cell
def _(bundle, gam_model, term_info):
    _feature_names = bundle["features"]
    _outputs = []

    _1d_items = []
    _2d_items = []
    _info_idx = 0
    for _i, _term in enumerate(gam_model.terms):
        if _term.isintercept:
            continue
        _info = term_info[_info_idx] if _info_idx < len(term_info) else {"name": f"term_{_i}", "features": []}
        if len(_info["features"]) >= 2:
            _2d_items.append((_i, _info))
        else:
            _1d_items.append((_i, _info))
        _info_idx += 1

    if _1d_items:
        _fig_1d, _axes = plt.subplots(
            1, len(_1d_items), figsize=(6 * len(_1d_items), 4), squeeze=False,
        )
        _axes = _axes[0]
        for _idx, (_term_i, _info) in enumerate(_1d_items):
            _feat_col = _info["features"][0]
            _XX = gam_model.generate_X_grid(term=_term_i)
            _pdep, _confi = gam_model.partial_dependence(
                term=_term_i, X=_XX, width=0.95,
            )
            _ax = _axes[_idx]
            _ax.plot(_XX[:, _feat_col], _pdep, label="PDP")
            _ax.fill_between(
                _XX[:, _feat_col], _confi[:, 0], _confi[:, 1],
                alpha=0.2, label="95% CI",
            )
            _ax.set_xlabel(_info["name"])
            _ax.set_ylabel("Częściowa zależność (log-rate)")
            _ax.set_title(f"PDP: {_info['name']}")
            _ax.legend()
        _fig_1d.tight_layout()
        _outputs.append(mo.md("## Wykresy Częściowej Zależności (PDP — 1D)"))
        _outputs.append(_fig_1d)

    for _term_i, _info in _2d_items:
        _n = 50
        _XX = gam_model.generate_X_grid(term=_term_i, n=_n)
        _Z = gam_model.partial_dependence(term=_term_i, X=_XX)
        _x1_u = np.unique(_XX[:, 0])
        _x2_u = np.unique(_XX[:, 1])
        _n1, _n2 = len(_x1_u), len(_x2_u)
        _X1 = _XX[:, 0].reshape(_n1, _n2)
        _X2 = _XX[:, 1].reshape(_n1, _n2)
        _Z_grid = _Z.reshape(_n1, _n2)

        _fig_2d, _ax_2d = plt.subplots(figsize=(7, 5))
        _contour = _ax_2d.contourf(_X1, _X2, _Z_grid, levels=20, cmap="coolwarm")
        _fig_2d.colorbar(_contour, ax=_ax_2d)
        _ax_2d.set_xlabel(_feature_names[0])
        _ax_2d.set_ylabel(_feature_names[1])
        _ax_2d.set_title("Heatmapa Interakcji (Tensor Effect)")
        _fig_2d.tight_layout()
        _outputs.append(mo.md("## Heatmapa Interakcji (Tensor Effect)"))
        _outputs.append(_fig_2d)

    mo.vstack(_outputs) if _outputs else mo.md("*Brak członów do wizualizacji PDP.*")
    return


@app.cell
def _(gam_model):
    def residuals_binning_plot(_X, _y, _label):
        _y_pred = gam_model.predict(_X)
        _residuals = _y - _y_pred
        _bins = pd.qcut(_X[:, 0], q=5, duplicates="drop")
        _res_df = pd.DataFrame({
            "team_baseline_lambda_bin": _bins,
            "residual": _residuals,
        })
        _agg = _res_df.groupby("team_baseline_lambda_bin", observed=True)["residual"].mean()

        _fig, _ax = plt.subplots(figsize=(7, 4))
        _agg.plot.bar(ax=_ax)
        _ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        _ax.set_ylabel("Średni residual (actual − predicted)")
        _ax.set_xlabel("Kwantyl team_baseline_lambda")
        _ax.set_title(f"Residuals Binning — {_label}")
        _ax.tick_params(axis="x", labelrotation=45)
        _fig.tight_layout()
        return _fig

    return (residuals_binning_plot,)


@app.cell
def _(bundle, residuals_binning_plot):
    _X_val = bundle["X_val"].to_numpy()
    _y_val = bundle["y_val"].to_numpy()

    _fig_val = residuals_binning_plot(_X_val, _y_val, "walidacja")

    mo.output.append(mo.md("## Wykres Residuów wg Kwantyli — Walidacja"))
    mo.output.append(_fig_val)
    return


@app.cell
def _(bundle, residuals_binning_plot):
    _X_train = bundle["X_train"].to_numpy()
    _y_train = bundle["y_train"].to_numpy()

    _fig_train = residuals_binning_plot(_X_train, _y_train, "trening")

    mo.output.append(mo.md("## Wykres Residuów wg Kwantyli — Trening"))
    mo.output.append(_fig_train)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 6. Ewaluacja
    """)
    return


@app.function
def evaluate_predictions(y_true_home, y_pred_home, y_true_away, y_pred_away):
    """Compute per-match Poisson deviance for home and away predictions.

    Returns means, standard errors of the mean (``SE = std(..., ddof=1) / sqrt(N)``),
    and ``Error_Vector``: Python list ``[dev_home for each match, then dev_away for each match]``.
    """

    def _poisson_deviance_per_sample(y_true, y_pred) -> np.ndarray:
        y_true = np.asarray(y_true, dtype=np.float64).ravel()
        y_pred = np.maximum(np.asarray(y_pred, dtype=np.float64).ravel(), 1e-15)
        safe_y_true = np.where(y_true > 0, y_true, 1.0)
        term1 = np.where(y_true > 0, y_true * np.log(safe_y_true / y_pred), 0.0)
        term2 = y_true - y_pred
        return 2.0 * (term1 - term2)

    dev_home = _poisson_deviance_per_sample(y_true_home, y_pred_home)
    dev_away = _poisson_deviance_per_sample(y_true_away, y_pred_away)
    n = int(dev_home.shape[0])
    if n == 0:
        raise ValueError("evaluate_predictions requires at least one match.")
    se_home = float(np.std(dev_home, ddof=1) / np.sqrt(n))
    se_away = float(np.std(dev_away, ddof=1) / np.sqrt(n))
    per_match_mean = (dev_home + dev_away) / 2.0
    se_mean = float(np.std(per_match_mean, ddof=1) / np.sqrt(n))
    err_vec = np.concatenate([dev_home, dev_away])
    return {
        "Deviance_home": round(float(np.mean(dev_home)), 4),
        "SE_home": round(se_home, 4),
        "Deviance_away": round(float(np.mean(dev_away)), 4),
        "SE_away": round(se_away, 4),
        "Deviance_mean": round(float(np.mean(per_match_mean)), 4),
        "SE_mean": round(se_mean, 4),
        "Error_Vector": err_vec.tolist(),
    }


@app.function
def compare_models_statistically(current_vector, best_vector, alpha=0.05):
    """Paired t-test on per-observation deviances (current vs best).

    Lower mean deviance is better. Returns ``comparison_status``:
    ``better_significant``, ``better_not_significant``, ``worse``, or ``error``.
    """
    a = np.asarray(current_vector, dtype=float).ravel()
    b = np.asarray(best_vector, dtype=float).ravel()
    if a.shape != b.shape or a.size == 0:
        return {
            "statistic": float("nan"),
            "pvalue": float("nan"),
            "alpha": alpha,
            "comparison_status": "error",
            "mean_current": float("nan"),
            "mean_best": float("nan"),
            "message": "Niezgodna długość wektorów lub pusty wektor.",
        }
    mean_c = float(np.mean(a))
    mean_b = float(np.mean(b))
    better = mean_c < mean_b
    tt = scipy.stats.ttest_rel(a, b)
    pval = float(tt.pvalue) if tt.pvalue is not None else float("nan")
    stat = float(tt.statistic)

    if not better:
        return {
            "statistic": stat,
            "pvalue": pval,
            "alpha": alpha,
            "comparison_status": "worse",
            "mean_current": mean_c,
            "mean_best": mean_b,
            "message": (
                "Bieżący model nie ma niższej średniej odchyleniowej Poissona "
                "niż najlepszy zapis w historii."
            ),
        }
    if pval < alpha:
        return {
            "statistic": stat,
            "pvalue": pval,
            "alpha": alpha,
            "comparison_status": "better_significant",
            "mean_current": mean_c,
            "mean_best": mean_b,
            "message": (
                "Bieżący model jest istotnie lepszy (p < α) od najlepszego zapisu w historii."
            ),
        }
    return {
        "statistic": stat,
        "pvalue": pval,
        "alpha": alpha,
        "comparison_status": "better_not_significant",
        "mean_current": mean_c,
        "mean_best": mean_b,
        "message": (
            "Średnia odchyleniowa jest niższa niż u najlepszego zapisu, "
            "ale różnica nie jest istotna statystycznie przy tym teście."
        ),
    }


@app.cell
def _(bundle, gam_model):
    _meta = bundle["val_long_meta"].copy()
    _meta["pred"] = gam_model.predict(bundle["X_val"].to_numpy())
    _h = _meta[_meta["is_home"] == 1][["match_id", "score", "pred"]].rename(
        columns={"score": "y_home", "pred": "pred_home"}
    )
    _a = _meta[_meta["is_home"] == 0][["match_id", "score", "pred"]].rename(
        columns={"score": "y_away", "pred": "pred_away"}
    )
    _eval_df = _h.merge(_a, on="match_id", how="inner")
    eval_metrics = evaluate_predictions(
        _eval_df["y_home"].to_numpy(),
        _eval_df["pred_home"].to_numpy(),
        _eval_df["y_away"].to_numpy(),
        _eval_df["pred_away"].to_numpy(),
    )

    _pair_rows = [
        ("Deviance_home", "SE_home"),
        ("Deviance_away", "SE_away"),
        ("Deviance_mean", "SE_mean"),
    ]
    _rows = [
        "| Metryka | Wartość (średnia ± SE) |",
        "|---------|------------------------|",
    ]
    for _dk, _sk in _pair_rows:
        _rows.append(
            f"| {_dk} | **{eval_metrics[_dk]} ± {eval_metrics[_sk]}** |"
        )

    mo.md("\n".join(_rows))
    return (eval_metrics,)


@app.cell
def _(eval_metrics):
    _log_path = os.path.join("logs", "gam_experiments_02_log.csv")

    mo.stop(
        not os.path.exists(_log_path),
        mo.md("*Brak historii GAM do porównania statystycznego.*"),
    )

    _hist = pd.read_csv(_log_path)
    _dm = pd.to_numeric(_hist["Deviance_mean"], errors="coerce")
    _best_idx = _dm.idxmin()
    _best_row = _hist.loc[_best_idx]
    _ev_raw = _best_row["Error_Vector"]

    try:
        _best_vec = json.loads(_ev_raw)
    except (json.JSONDecodeError, TypeError):
        _best_vec = None

    mo.stop(
        _best_vec is None,
        mo.md("**Niepoprawny format Error_Vector w historii.**"),
    )

    _result = compare_models_statistically(
        eval_metrics["Error_Vector"],
        _best_vec,
        alpha=0.05,
    )

    _status = _result.get("comparison_status", "error")

    mo.stop(
        _status == "error",
        mo.md(f"**{_result.get('message', 'Błąd porównania')}**"),
    )

    _emoji_map = {
        "better_significant": "🟢",
        "better_not_significant": "🟠",
        "worse": "🔴",
    }
    _emoji = _emoji_map.get(_status, "—")

    _p = _result.get("pvalue")
    _p_str = f"{_p:.4g}" if _p is not None and not (isinstance(_p, float) and np.isnan(_p)) else "n/a"

    mo.md(
        f"### Porównanie statystyczne z najlepszym zapisem w historii\n\n"
        f"**{_emoji}** `comparison_status={_status}`\n\n"
        f"- **p-wartość** (test t dla par): **{_p_str}**\n"
        f"- **Średnia bieżąca / średnia najlepszego zapisu**: "
        f"{_result.get('mean_current', float('nan')):.6g} / "
        f"{_result.get('mean_best', float('nan')):.6g}\n\n"
        f"{_result.get('message', '')}"
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 7. Dziennik Eksperymentów
    """)
    return


@app.cell
def _():
    get_save_clicks, set_save_clicks = mo.state(0)
    return get_save_clicks, set_save_clicks


@app.cell
def _():
    experiment_notes = mo.ui.text_area(label="Notatki do tego eksperymentu:")
    save_button = mo.ui.button(
        value=0,
        on_click=lambda value: value + 1,
        label="Zapisz wynik do dziennika",
    )
    mo.vstack([experiment_notes, save_button])
    return experiment_notes, save_button


@app.cell(hide_code=True)
def _(
    eval_metrics,
    experiment_notes,
    formula_str,
    gam_model,
    get_save_clicks,
    save_button,
    set_save_clicks,
    training_time,
):
    _log_file = os.path.join("logs", "gam_experiments_02_log.csv")
    os.makedirs(os.path.dirname(_log_file), exist_ok=True)

    if save_button.value is not None and save_button.value > get_save_clicks():
        set_save_clicks(save_button.value)

        _notes = experiment_notes.value
        if _notes is not None and str(_notes).strip() != "":
            _metrics_row = {
                k: v
                for k, v in eval_metrics.items()
                if k != "Error_Vector"
            }
            _metrics_row["Error_Vector"] = json.dumps(
                eval_metrics["Error_Vector"],
                ensure_ascii=False,
            )
            _new_data = {
                "Data": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"),
                "Formula": formula_str,
                "Lambda": str(gam_model.lam),
                "Training_Time_s": training_time,
                **_metrics_row,
                "Notatki": _notes,
            }
            _new_entry = pd.DataFrame([_new_data])

            _should_save = True
            if os.path.exists(_log_file):
                _existing_df = pd.read_csv(_log_file)
                _cols_to_compare = [
                    c
                    for c in _new_entry.columns
                    if c not in ("Data", "Error_Vector") and c in _existing_df.columns
                ]
                if (
                    _cols_to_compare
                    and len(_existing_df) > 0
                    and _existing_df[_cols_to_compare]
                    .iloc[-1:]
                    .reset_index(drop=True)
                    .equals(
                        _new_entry[_cols_to_compare].reset_index(drop=True)
                    )
                ):
                    _should_save = False

            if _should_save:
                if os.path.exists(_log_file):
                    _combined = pd.concat(
                        [_existing_df, _new_entry], ignore_index=True
                    )
                    _combined.to_csv(_log_file, index=False)
                else:
                    _new_entry.to_csv(_log_file, index=False)

    if os.path.exists(_log_file):
        _df_history = pd.read_csv(_log_file)
        _df_display = _df_history.drop(columns=["Error_Vector"], errors="ignore")
        _output = mo.ui.table(_df_display, selection=None)
    else:
        _output = mo.md("*Brak historii eksperymentów. Zapisz pierwszy wynik!*")

    _output
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # 8. Dobór `bias_correction` (siatka na sezonie walidacji)

    Poniżej w pętli budowane są tymczasowe ramki z różnym mnożnikiem `bias_correction`
    przy tych samych prawdopodobieństwach z kursów.
    Dla każdej wartości liczona jest ta sama metryka co przy ewaluacji GAM (`Deviance_mean` na walidacji),
    żeby wskazać **jedną skalę całkowitego λ** zamiast dopasowywać osobną regresję liniową korekty.
    Dodatkowo, we wcześniejszych eksperymentach najlepsze wyniki osiągała regresja z jedną cechą lambdy drużyny,
    więc tutaj sprawdzamy, czy można dobrać parametr `bias_correction` tak, aby uzyskać równie dobry wynik.
    """)
    return


@app.cell
def _(df_wide_probs):
    _season_val = "2024/2025"
    _bias_grid = np.linspace(0.96, 1.08, 49)
    _rows = []
    for _bc in _bias_grid:
        _df_temp = add_baseline_poisson_lambdas(
            df_wide_probs,
            prob_home_col="prob_trimmed_avg_1",
            prob_away_col="prob_trimmed_avg_2",
            prob_over25_col="prob_trimmed_avg_over_25",
            bias_correction=float(_bc),
        )
        df_val_temp = _df_temp.loc[
            _df_temp["season"] == _season_val,
            [
                "home_score",
                "away_score",
                "baseline_lambda_home",
                "baseline_lambda_away",
            ],
        ].dropna()
        y_true_home = df_val_temp["home_score"].to_numpy(dtype=np.float64)
        y_true_away = df_val_temp["away_score"].to_numpy(dtype=np.float64)
        _em = evaluate_predictions(
            y_true_home,
            df_val_temp["baseline_lambda_home"].to_numpy(dtype=np.float64),
            y_true_away,
            df_val_temp["baseline_lambda_away"].to_numpy(dtype=np.float64),
        )
        _rows.append(
            {
                "bias_correction": float(_bc),
                "Deviance_mean": _em["Deviance_mean"],
                "SE_mean": _em["SE_mean"],
                "Deviance_home": _em["Deviance_home"],
                "Deviance_away": _em["Deviance_away"],
                "n_matches": len(df_val_temp),
            }
        )

    bias_sweep_df = pd.DataFrame(_rows)
    _best_i = int(bias_sweep_df["Deviance_mean"].idxmin())
    best_bias_correction = float(bias_sweep_df.loc[_best_i, "bias_correction"])
    best_deviance_mean = float(bias_sweep_df.loc[_best_i, "Deviance_mean"])

    _fig, _ax = plt.subplots(figsize=(8, 4))
    _ax.plot(
        bias_sweep_df["bias_correction"],
        bias_sweep_df["Deviance_mean"],
        marker="o",
        markersize=3,
        label="Deviance_mean (walidacja)",
    )
    _ax.axvline(
        best_bias_correction,
        color="C1",
        linestyle="--",
        linewidth=1.2,
        label=f"Minimum @ bias_correction = {best_bias_correction:.4f}",
    )
    _ax.scatter(
        [best_bias_correction],
        [best_deviance_mean],
        color="C1",
        s=80,
        zorder=5,
        label=f"Najlepsza: {best_deviance_mean:.4f}",
    )
    _ax.set_xlabel("bias_correction")
    _ax.set_ylabel("Deviance_mean")
    _ax.set_title(
        "Poisson deviance (baseline λ z kursów) vs. mnożnik bias_correction — walidacja"
    )
    _ax.legend(loc="best", fontsize=8)
    _fig.tight_layout()

    mo.vstack(
        [
            mo.md(
                f"**Najlepszy `bias_correction` (min. Deviance_mean):** "
                f"`{best_bias_correction:.4f}` → Deviance_mean = **{best_deviance_mean:.4f}** "
                f"(N = {int(bias_sweep_df.loc[_best_i, 'n_matches'])} meczów)."
            ),
            _fig,
            mo.md(
                "*Uwaga:* to jest kalibracja **jednym skalarem** na całkowite oczekiwane gole; "
                "regresja liniowa na predykcjach daje bogatszą korektę, ale siatka po "
                "`bias_correction` bywa wystarczająca do wyboru domyślnej wartości w pipeline."
            ),
        ]
    )
    return


@app.cell
def _(df_wide_full, gam_model, opp_term_type, team_term_type, use_interaction):
    """Kalibracja ``exp(B0 + B1 * λ)`` z pierwszego i ostatniego współczynnika ``coef_`` (jak w prostym ``l(0)``)."""
    _formula_is_l0_only = (
        team_term_type.value == "Liniowa (l)"
        and opp_term_type.value == "Brak"
        and not use_interaction.value
    )
    mo.stop(
        not _formula_is_l0_only,
        mo.md(
            "*Ta sekcja uruchamia się tylko przy formule **wyłącznie `l(0)`**: "
            "Team Lambda → **Liniowa (l)**, Opponent Lambda → **Brak**, "
            "bez zaznaczonej interakcji `te`. Wtedy `coef_` to intercept + jeden nachylenie "
            "dla `team_baseline_lambda`.*"
        ),
    )
    mo.stop(
        len(gam_model.coef_) != 2,
        mo.md(
            f"*Oczekiwano 2 współczynników (intercept + `l(0)`), jest **{len(gam_model.coef_)}**. "
            "Sprawdź formułę i ponów trening.*"
        ),
    )

    BETA_0 = float(gam_model.coef_[-1])
    BETA_1 = float(gam_model.coef_[0])

    def correct_lambdas(base_lambda_series):
        # Równanie: e^(B0 + B1 * lambda)
        x = np.asarray(base_lambda_series, dtype=np.float64)
        return np.exp(BETA_0 + BETA_1 * x)

    _season_val = "2024/2025"
    _df_val = df_wide_full.loc[
        df_wide_full["season"] == _season_val,
        ["home_score", "away_score", "baseline_lambda_home", "baseline_lambda_away"],
    ].dropna()

    _y_true_home = _df_val["home_score"].to_numpy(dtype=np.float64)
    _y_true_away = _df_val["away_score"].to_numpy(dtype=np.float64)
    pred_home = correct_lambdas(_df_val["baseline_lambda_home"])
    pred_away = correct_lambdas(_df_val["baseline_lambda_away"])

    eval_linear_lambda_cal = evaluate_predictions(
        _y_true_home,
        pred_home,
        _y_true_away,
        pred_away,
    )

    _pair_rows = [
        ("Deviance_home", "SE_home"),
        ("Deviance_away", "SE_away"),
        ("Deviance_mean", "SE_mean"),
    ]
    _table_lines = [
        "| Metryka | Wartość (średnia ± SE) |",
        "|---------|------------------------|",
    ]
    for _dk, _sk in _pair_rows:
        _table_lines.append(
            f"| {_dk} | **{eval_linear_lambda_cal[_dk]} ± {eval_linear_lambda_cal[_sk]}** |"
        )

    _parts = [
        mo.md("## Korekta `exp(B0 + B1·λ)` ze współczynników GAM — walidacja"),
        mo.md(
            f"`BETA_0` (``coef_[-1]``, intercept) = **{BETA_0:.6g}** · "
            f"`BETA_1` (``coef_[0]``, slope) = **{BETA_1:.6g}** · "
            f"N meczów = **{len(_df_val)}**"
        ),
        mo.md(
            "```text\n"
            + np.array2string(
                np.asarray(gam_model.coef_, dtype=float),
                precision=6,
                separator=", ",
            )
            + "\n```\n*Pełny wektor* ``gam_model.coef_``*.*"
        ),
        mo.md("\n".join(_table_lines)),
    ]

    mo.vstack(_parts)
    return BETA_0, BETA_1, correct_lambdas


@app.cell
def _(
    BETA_0,
    BETA_1,
    correct_lambdas,
    df_wide_full,
    gam_model,
    opp_term_type,
    team_term_type,
    use_interaction,
):
    """Ten sam warunek co sekcja kalibracji; wykres nie zależy od jej wyjść (gdy `mo.stop`, brak `return`)."""
    _formula_is_l0_only = (
        team_term_type.value == "Liniowa (l)"
        and opp_term_type.value == "Brak"
        and not use_interaction.value
    )
    mo.stop(
        not _formula_is_l0_only,
        mo.md(
            "*Wykres kalibracji — dostępny tylko przy tej samej formule co sekcja powyżej "
            "(wyłącznie `l(0)` na Team Lambda).*"
        ),
    )
    mo.stop(
        len(gam_model.coef_) != 2,
        mo.md("*Brak wykresu:* model nie ma dokładnie 2 współczynników (intercept + `l(0)`).*"),
    )

    _season_val = "2024/2025"
    _v = df_wide_full.loc[df_wide_full["season"] == _season_val]
    _lam_h = _v["baseline_lambda_home"].dropna().to_numpy(dtype=np.float64)
    _lam_a = _v["baseline_lambda_away"].dropna().to_numpy(dtype=np.float64)
    _lam_all = np.concatenate([_lam_h, _lam_a])
    _lo, _hi = float(np.min(_lam_all)), float(np.max(_lam_all))
    _pad = 0.05 * (_hi - _lo) if _hi > _lo else 0.1
    _xs = np.linspace(_lo - _pad, _hi + _pad, 300)
    _ys = correct_lambdas(_xs)

    _fig, _ax = plt.subplots(figsize=(7, 4))
    _ax.plot(_xs, _ys, color="C0", label=r"$\lambda_{\mathrm{out}} = e^{B_0 + B_1 \lambda}$")
    _ax.scatter(_lam_h, correct_lambdas(_lam_h), s=12, alpha=0.35, color="C1", label="Walidacja: home")
    _ax.scatter(_lam_a, correct_lambdas(_lam_a), s=12, alpha=0.35, color="C2", label="Walidacja: away")
    _ax.plot([_lo - _pad, _hi + _pad], [_lo - _pad, _hi + _pad], "k--", alpha=0.25, linewidth=0.8, label="y = x")
    _ax.set_xlabel("baseline λ (wejście)")
    _ax.set_ylabel("skorygowane λ (wyjście)")
    _ax.set_title(f"Korekta exp(B0 + B1·λ)  ·  B0={BETA_0:.4g}, B1={BETA_1:.4g}")
    _ax.legend(loc="best", fontsize=8)
    _ax.grid(True, alpha=0.3)
    _fig.tight_layout()

    mo.vstack([
        mo.md("### Wykres funkcji `correct_lambdas` (na zakresie λ ze zbioru walidacyjnego)"),
        _fig,
    ])
    return


if __name__ == "__main__":
    app.run()
