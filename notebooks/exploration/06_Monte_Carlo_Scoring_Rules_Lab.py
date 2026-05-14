import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")

with app.setup:
    from pathlib import Path
    import sys

    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from scipy.stats import poisson

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    from src.models.components import (
        PoissonMatrixBuilder,
        ExpectedPointsOptimizer,
        ExpectedPointsRule,
    )


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    # Monte Carlo: Reguły Punktacji i Pułapki Optymalizacji

    _Ścieżka w repozytorium:_ `notebooks/exploration/06_Monte_Carlo_Scoring_Rules_Lab.py`

    ## Cel

    Ten notebook jest eksperymentem Monte Carlo, którego celem jest **matematyczne
    wykazanie** trzech kluczowych problemów związanych z optymalizacją modeli
    predykcyjnych w konkursach piłkarskich typu Supertyper:

    1. **Metryka ważona (Weighted NLL) nie jest regułą właściwą (proper scoring rule)** —
       optymalizator może ją „obejść", drastycznie deformując rozkład
       prawdopodobieństwa zamiast poprawnie go kalibrować.
    2. **Dyskretna metryka punktowa (średnie punkty)** na krótkich próbkach daje
       niestabilne optimum, które zmienia się z sezonu na sezon.
    3. **Misspecyfikacja modelu** (np. asymetryczny bias faworytów i outsiderów)
       sprawia, że metryka punktowa i NLL prowadzą do zupełnie różnych
       parametrów.

    Wszystkie eksperymenty opierają się na **danych syntetycznych** z kontrolowanym
    procesem generującym (DGP), dzięki czemu znamy prawdziwe parametry i możemy
    precyzyjnie ocenić, co dany optymalizator „znajduje".

    ---

    ### Zasady punktacji Supertyper

    | Trafienie | Punkty |
    |---|---|
    | Dokładny wynik | **3** |
    | Poprawna różnica bramek | **2** |
    | Poprawne rozstrzygnięcie 1X2 | **1** |
    | Pudło | **0** |
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ## Tło Matematyczne

    ### Model Dixon-Colesa

    Bazowy model zakłada, że gole gospodarzy ($X$) i gości ($Y$) mają rozkład
    Poissona z parametrami $\lambda_h$ i $\lambda_a$. Korekta Dixona-Colesa
    modyfikuje cztery najniższe wyniki za pomocą parametru $\rho$:

    $$P_{DC}(X\!=\!x, Y\!=\!y)
      = P_{\text{Poisson}}(x;\lambda_h) \;\cdot\; P_{\text{Poisson}}(y;\lambda_a)
        \;\cdot\; \tau(x,y,\lambda_h,\lambda_a,\rho)$$

    gdzie:

    $$\tau(x,y) = \begin{cases}
    1 - \lambda_h \lambda_a \rho & (0{:}0) \\
    1 + \lambda_h \rho            & (0{:}1) \\
    1 + \lambda_a \rho            & (1{:}0) \\
    1 - \rho                      & (1{:}1) \\
    1                             & \text{w.p.p.}
    \end{cases}$$

    ### Reguły właściwe (proper scoring rules)

    **Reguła właściwa** to funkcja kosztu minimalizowana wtedy i tylko wtedy, gdy
    model raportuje swoje prawdziwe przekonania. Standardowe NLL jest regułą
    właściwą:

    $$\text{NLL} = -\frac{1}{N}\sum_{i=1}^{N}\log P_\theta(h_i, a_i)$$

    **Ważone NLL** nie jest regułą właściwą, ponieważ agreguje prawdopodobieństwo
    z wagami zależnymi od typu trafienia. Optymalizator może uzyskać niższe
    ważone NLL nie poprawiając kalibracji, lecz deformując rozkład — przenosząc
    masę na wyniki, które dają najwyższą wagę.

    ### Parametry modelu

    - `bias_correction` — globalny mnożnik obu lambd:
      $\lambda_h^{\text{model}} = \text{bias} \cdot \lambda_h^{\text{base}}$,
      $\lambda_a^{\text{model}} = \text{bias} \cdot \lambda_a^{\text{base}}$.
    - `rho` — parametr korekty Dixona-Colesa dla niskich wyników.
    """)
    return


@app.cell
def _():
    is_script_mode = mo.app_meta().mode == "script"
    return


@app.cell
def _():
    MAX_GOALS = 10
    SEED = 20260514
    TRUE_BIAS = 1.0
    TRUE_RHO = -0.15
    return MAX_GOALS, SEED, TRUE_BIAS, TRUE_RHO


@app.function
def simulate_matches(
    n_matches, rng, true_rho, true_bias, max_goals,
    lambda_lo=0.8, lambda_hi=2.5,
):
    """Generate synthetic Dixon-Coles football matches with known params."""
    base_home = rng.uniform(lambda_lo, lambda_hi, size=n_matches)
    base_away = rng.uniform(lambda_lo, lambda_hi, size=n_matches)
    true_home = base_home * true_bias
    true_away = base_away * true_bias

    builder = PoissonMatrixBuilder(rho=true_rho, max_goals_matrix=max_goals)
    home_scores = np.empty(n_matches, dtype=int)
    away_scores = np.empty(n_matches, dtype=int)
    for i in range(n_matches):
        mat = builder.build_matrix(float(true_home[i]), float(true_away[i]))
        flat = rng.choice(mat.size, p=mat.ravel())
        home_scores[i], away_scores[i] = np.unravel_index(flat, mat.shape)

    return {
        "base_home": base_home,
        "base_away": base_away,
        "true_home": true_home,
        "true_away": true_away,
        "home_score": home_scores,
        "away_score": away_scores,
    }


@app.function
def build_matrices_batch(lambda_h, lambda_a, rho, max_goals):
    """Build Dixon-Coles probability matrices for all matches at once.

    Returns ndarray of shape ``(n, size, size)`` where ``size = max_goals + 1``.
    """
    lh = np.asarray(lambda_h, dtype=np.float64)
    la = np.asarray(lambda_a, dtype=np.float64)
    n = lh.shape[0]
    size = max_goals + 1
    goals = np.arange(size)

    p_home = poisson.pmf(goals[None, :], lh[:, None])
    p_away = poisson.pmf(goals[None, :], la[:, None])
    matrices = p_home[:, :, None] * p_away[:, None, :]

    matrices[:, 0, 0] *= 1.0 - lh * la * rho
    matrices[:, 0, 1] *= 1.0 + lh * rho
    matrices[:, 1, 0] *= 1.0 + la * rho
    matrices[:, 1, 1] *= 1.0 - rho

    np.clip(matrices, 0.0, None, out=matrices)
    sums = matrices.sum(axis=(1, 2), keepdims=True)
    sums = np.where(sums > 0, sums, 1.0)
    matrices /= sums
    return matrices


@app.function
def score_predictions_vectorized(pred_h, pred_a, actual_h, actual_a):
    """Vectorized Supertyper scoring for arrays of predictions."""
    exact = (pred_h == actual_h) & (pred_a == actual_a)
    diff_ok = (pred_h - pred_a) == (actual_h - actual_a)
    sign_ok = np.sign(pred_h - pred_a) == np.sign(actual_h - actual_a)
    return np.where(exact, 3, np.where(diff_ok, 2, np.where(sign_ok, 1, 0)))


@app.cell
def _():
    def compute_nll_grid(
        base_h, base_a, actual_h, actual_a, bias_vals, rho_vals, max_goals,
    ):
        """Average NLL on a 2-D (bias x rho) grid — vectorized."""
        grid = np.full((len(bias_vals), len(rho_vals)), np.nan)
        ah = np.asarray(actual_h, dtype=int)
        aa = np.asarray(actual_a, dtype=int)
        idx = np.arange(len(ah))
        for i, b in enumerate(bias_vals):
            adj_h = base_h * b
            adj_a = base_a * b
            for j, r in enumerate(rho_vals):
                mats = build_matrices_batch(adj_h, adj_a, float(r), max_goals)
                probs = np.clip(mats[idx, ah, aa], 1e-15, None)
                grid[i, j] = -np.log(probs).mean()
        return grid

    def compute_wnll_grid(
        base_h, base_a, actual_h, actual_a, bias_vals, rho_vals, max_goals,
        exact_w=1.0, diff_w=2.0 / 3.0, outcome_w=1.0 / 3.0,
    ):
        """Average weighted NLL on a 2-D grid — vectorized."""
        grid = np.full((len(bias_vals), len(rho_vals)), np.nan)
        ah = np.asarray(actual_h, dtype=int)
        aa = np.asarray(actual_a, dtype=int)
        n = len(ah)
        size = max_goals + 1
        idx = np.arange(n)
        goals = np.arange(size)
        gdiff = goals[:, None] - goals[None, :]
        gsign = np.sign(gdiff)

        actual_diff = ah - aa
        actual_sign = np.sign(actual_diff)

        for i, b in enumerate(bias_vals):
            adj_h = base_h * b
            adj_a = base_a * b
            for j, r in enumerate(rho_vals):
                mats = build_matrices_batch(adj_h, adj_a, float(r), max_goals)

                exact_probs = mats[idx, ah, aa]

                diff_mask = gdiff[None, :, :] == actual_diff[:, None, None]
                exact_ind = np.zeros((n, size, size), dtype=bool)
                exact_ind[idx, ah, aa] = True
                diff_mask &= ~exact_ind
                diff_probs = (mats * diff_mask).sum(axis=(1, 2))

                out_mask = gsign[None, :, :] == actual_sign[:, None, None]
                out_mask &= ~diff_mask & ~exact_ind
                out_probs = (mats * out_mask).sum(axis=(1, 2))

                weighted = (
                    exact_w * exact_probs
                    + diff_w * diff_probs
                    + outcome_w * out_probs
                )
                weighted = np.clip(weighted, 1e-15, None)
                grid[i, j] = -np.log(weighted).mean()
        return grid

    def compute_pts_grid(
        base_h, base_a, actual_h, actual_a, bias_vals, rho_vals, max_goals,
    ):
        """Average Supertyper points on a 2-D grid — vectorized."""
        pred_cap = min(4, max_goals)
        rules = ExpectedPointsRule(exact=3, goal_diff=2, outcome=1, miss=0)
        optimizer = ExpectedPointsOptimizer(
            rules=rules,
            max_goals_prediction=pred_cap,
            max_goals_matrix=max_goals,
        )
        pts_tensor = optimizer._points_matrix
        pred_size = pred_cap + 1

        grid = np.zeros((len(bias_vals), len(rho_vals)))
        ah = np.asarray(actual_h, dtype=int)
        aa = np.asarray(actual_a, dtype=int)
        n = len(ah)

        for j, r in enumerate(rho_vals):
            for i, b in enumerate(bias_vals):
                mats = build_matrices_batch(
                    base_h * b, base_a * b, float(r), max_goals,
                )
                ep = np.einsum("ijkl,mkl->mij", pts_tensor, mats)
                flat_best = np.argmax(ep.reshape(n, -1), axis=1)
                ph = flat_best // pred_size
                pa = flat_best % pred_size
                grid[i, j] = score_predictions_vectorized(
                    ph, pa, ah, aa,
                ).mean()
        return grid

    return compute_nll_grid, compute_pts_grid, compute_wnll_grid


@app.function
def plot_metric_heatmap(
    ax, grid, bias_vals, rho_vals, title, true_bias, true_rho,
    *,
    minimize=True,
    cmap="viridis",
):
    """Plot a 2-D metric surface with true and optimal parameters."""
    R, B = np.meshgrid(rho_vals, bias_vals)
    valid = ~np.isnan(grid)
    if not valid.any():
        ax.set_title(title + " (brak danych)")
        return

    if minimize:
        best_flat = int(np.nanargmin(grid))
        cmap_used = cmap + "_r"
    else:
        best_flat = int(np.nanargmax(grid))
        cmap_used = cmap

    best_idx = np.unravel_index(best_flat, grid.shape)
    opt_bias = float(bias_vals[best_idx[0]])
    opt_rho = float(rho_vals[best_idx[1]])

    cf = ax.contourf(R, B, grid, levels=20, cmap=cmap_used, alpha=0.92)
    ax.figure.colorbar(cf, ax=ax, shrink=0.82, pad=0.03)

    ax.scatter(
        true_rho, true_bias, marker="*", s=300, c="red",
        edgecolors="white", linewidths=1.5, zorder=5,
        label=f"Prawdziwe ({true_bias:.2f}, {true_rho:.2f})",
    )
    ax.scatter(
        opt_rho, opt_bias, marker="X", s=220, c="lime",
        edgecolors="black", linewidths=1.5, zorder=5,
        label=f"Optimum ({opt_bias:.2f}, {opt_rho:.2f})",
    )

    ax.set_xlabel(r"$\rho$")
    ax.set_ylabel("bias_correction")
    ax.set_title(title, fontsize=11, pad=8)
    ax.legend(fontsize=7, loc="upper right")


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Eksperyment A: Odtwarzanie Parametrów i „Hackowanie" Metryki

    Generujemy **5 000 meczów** ze znanych parametrów (`true_bias = 1.0`,
    `true_rho = −0.15`). Następnie przeszukujemy siatkę parametrów pod kątem
    trzech metryk:

    1. **Standard NLL** — prawidłowa reguła punktacji (proper scoring rule).
    2. **Weighted NLL** — wagi: dokładny = 1.0, różnica = 0.66,
       rozstrzygnięcie = 0.33.
    3. **Średnie punkty** — zrealizowane punkty konkursowe po wyborze
       optymalnego typu (3/2/1/0).

    **Hipoteza:** Standard NLL poprawnie odtworzy prawdziwe parametry, a
    Weighted NLL „zhackuje" metrykę, przesuwając $\rho$ do skrajnie dodatnich
    wartości i `bias` w dół.
    """)
    return


@app.cell
def _(MAX_GOALS, SEED, TRUE_BIAS, TRUE_RHO):
    rng_a = np.random.default_rng(SEED)
    data_a = simulate_matches(5_000, rng_a, TRUE_RHO, TRUE_BIAS, MAX_GOALS)
    mo.md(
        f"**Eksperyment A:** wygenerowano **{len(data_a['home_score'])}** meczów. "
        f"Średnie gole: {data_a['home_score'].mean():.2f} (gosp.) / "
        f"{data_a['away_score'].mean():.2f} (gości)."
    )
    return (data_a,)


@app.cell
def _(
    MAX_GOALS,
    compute_nll_grid,
    compute_pts_grid,
    compute_wnll_grid,
    data_a,
):
    bias_grid_a = np.round(np.arange(0.70, 1.301, 0.05), 3)
    rho_grid_a = np.round(np.arange(-0.30, 0.201, 0.05), 3)

    nll_a = compute_nll_grid(
        data_a["base_home"], data_a["base_away"],
        data_a["home_score"], data_a["away_score"],
        bias_grid_a, rho_grid_a, MAX_GOALS,
    )
    wnll_a = compute_wnll_grid(
        data_a["base_home"], data_a["base_away"],
        data_a["home_score"], data_a["away_score"],
        bias_grid_a, rho_grid_a, MAX_GOALS,
    )
    pts_a = compute_pts_grid(
        data_a["base_home"], data_a["base_away"],
        data_a["home_score"], data_a["away_score"],
        bias_grid_a, rho_grid_a, MAX_GOALS,
    )
    return bias_grid_a, nll_a, pts_a, rho_grid_a, wnll_a


@app.cell
def _(TRUE_BIAS, TRUE_RHO, bias_grid_a, nll_a, pts_a, rho_grid_a, wnll_a):
    fig_a, axes_a = plt.subplots(1, 3, figsize=(20, 5.5))

    plot_metric_heatmap(
        axes_a[0], nll_a, bias_grid_a, rho_grid_a,
        "Standard NLL (niższe = lepsze)",
        TRUE_BIAS, TRUE_RHO, minimize=True,
    )
    plot_metric_heatmap(
        axes_a[1], wnll_a, bias_grid_a, rho_grid_a,
        "Weighted NLL (niższe = lepsze)",
        TRUE_BIAS, TRUE_RHO, minimize=True,
    )
    plot_metric_heatmap(
        axes_a[2], pts_a, bias_grid_a, rho_grid_a,
        "Średnie punkty (wyższe = lepsze)",
        TRUE_BIAS, TRUE_RHO, minimize=False,
    )

    fig_a.suptitle(
        "Eksperyment A: Porównanie trzech metryk na siatce (bias, ρ)",
        fontsize=13, y=1.02,
    )
    fig_a.tight_layout()
    fig_a
    return


@app.cell
def _(TRUE_BIAS, TRUE_RHO, bias_grid_a, nll_a, pts_a, rho_grid_a, wnll_a):
    def _best(grid, minimize):
        fn = np.nanargmin if minimize else np.nanargmax
        idx = np.unravel_index(int(fn(grid)), grid.shape)
        return float(bias_grid_a[idx[0]]), float(rho_grid_a[idx[1]]), float(grid[idx])

    nll_b, nll_r, nll_v = _best(nll_a, True)
    wnll_b, wnll_r, wnll_v = _best(wnll_a, True)
    pts_b, pts_r, pts_v = _best(pts_a, False)

    summary_a = pd.DataFrame([
        {"Metryka": "Standard NLL", "bias*": nll_b, "ρ*": nll_r,
         "Wartość": f"{nll_v:.4f}", "Δbias": f"{nll_b - TRUE_BIAS:+.2f}",
         "Δρ": f"{nll_r - TRUE_RHO:+.2f}"},
        {"Metryka": "Weighted NLL", "bias*": wnll_b, "ρ*": wnll_r,
         "Wartość": f"{wnll_v:.4f}", "Δbias": f"{wnll_b - TRUE_BIAS:+.2f}",
         "Δρ": f"{wnll_r - TRUE_RHO:+.2f}"},
        {"Metryka": "Średnie punkty", "bias*": pts_b, "ρ*": pts_r,
         "Wartość": f"{pts_v:.4f}", "Δbias": f"{pts_b - TRUE_BIAS:+.2f}",
         "Δρ": f"{pts_r - TRUE_RHO:+.2f}"},
    ])
    mo.vstack([
        mo.md("### Podsumowanie Eksperymentu A"),
        mo.ui.table(summary_a),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Interpretacja Eksperymentu A

    **Standard NLL** poprawnie lokalizuje optimum w okolicach prawdziwych
    parametrów (`bias ≈ 1.0`, `ρ ≈ −0.15`). To dlatego, że NLL jest
    **regułą właściwą (proper scoring rule)**: jej minimum matematycznie
    pokrywa się z prawdziwym rozkładem generującym dane.

    **Weighted NLL** drastycznie odchyla się od prawdy. Optymalizator „odkrywa",
    że:

    - obniżając `bias` zmniejsza łączną liczbę bramek, przez co masa
      prawdopodobieństwa skupia się na niskich wynikach (0:0, 1:0, 0:1, 1:1);
    - podnosząc `ρ` do wartości dodatnich dodatkowo przenosi masę z remisów
      (0:0, 1:1) na minimalne wygrane (1:0, 0:1), które są najczęściej
      typowanymi wynikami.

    W efekcie model przypisuje ogromną wagę do kilku wąskich scenariuszy — co
    obniża ważoną stratę, ale **całkowicie niszczy kalibrację**. Weighted NLL
    nie jest regułą właściwą, więc nic nie powstrzymuje optymalizatora przed
    takim „hackingiem".

    **Średnie punkty** lokalizują optimum blisko prawdziwych parametrów, ale
    powierzchnia tej metryki jest wyraźnie bardziej zaszumiona niż NLL.
    Na 5 000 meczów wynik wygląda jeszcze sensownie — Eksperyment B pokaże,
    co się dzieje na mniejszych próbkach.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Eksperyment B: Niestabilność Metryki Punktowej

    Symulujemy **100 niezależnych „sezonów"**, każdy po **300 meczów**,
    generowanych z tych samych prawdziwych parametrów. Dla każdego sezonu
    szukamy pary `(bias, ρ)`, która **maksymalizuje średnie punkty**.

    **Hipoteza:** optymalne parametry będą mocno się różnić między sezonami,
    co dowodzi, że optymalizacja ciągłego rozkładu pod dyskretną metrykę
    punktową na krótkich próbkach prowadzi do masowego overfittingu do szumu.
    """)
    return


@app.cell
def _(MAX_GOALS, SEED, TRUE_BIAS, TRUE_RHO, compute_pts_grid):
    rng_b = np.random.default_rng(SEED + 1_000)
    n_seasons_b = 100
    n_per_season_b = 300
    bias_grid_b = np.round(np.arange(0.70, 1.301, 0.05), 3)
    rho_grid_b = np.round(np.arange(-0.30, 0.201, 0.05), 3)

    season_rows = []
    for s in range(n_seasons_b):
        season_data = simulate_matches(
            n_per_season_b, rng_b, TRUE_RHO, TRUE_BIAS, MAX_GOALS,
        )
        grid = compute_pts_grid(
            season_data["base_home"], season_data["base_away"],
            season_data["home_score"], season_data["away_score"],
            bias_grid_b, rho_grid_b, MAX_GOALS,
        )
        best_idx = np.unravel_index(int(np.argmax(grid)), grid.shape)
        season_rows.append({
            "season": s,
            "best_bias": float(bias_grid_b[best_idx[0]]),
            "best_rho": float(rho_grid_b[best_idx[1]]),
            "avg_pts": float(grid[best_idx]),
        })

    df_optima_b = pd.DataFrame(season_rows)
    return (df_optima_b,)


@app.cell
def _(TRUE_BIAS, TRUE_RHO, df_optima_b):
    fig_b, ax_b = plt.subplots(figsize=(8, 6))

    ax_b.scatter(
        df_optima_b["best_rho"],
        df_optima_b["best_bias"],
        alpha=0.55, s=50, edgecolors="navy", linewidths=0.5,
        label="Optimum sezonu",
    )
    ax_b.scatter(
        TRUE_RHO, TRUE_BIAS, marker="*", s=400, c="red",
        edgecolors="white", linewidths=1.5, zorder=5,
        label=f"Prawdziwe ({TRUE_BIAS:.2f}, {TRUE_RHO:.2f})",
    )

    mean_b = df_optima_b["best_bias"].mean()
    mean_r = df_optima_b["best_rho"].mean()
    ax_b.axhline(mean_b, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax_b.axvline(mean_r, color="gray", ls="--", lw=0.8, alpha=0.6)

    ax_b.set_xlabel(r"Najlepsze $\rho$")
    ax_b.set_ylabel("Najlepszy bias_correction")
    ax_b.set_title(
        f"Eksperyment B: Optymalne parametry w {len(df_optima_b)} sezonach "
        f"(po {300} meczów)",
        fontsize=11,
    )
    ax_b.legend(fontsize=9)
    ax_b.grid(alpha=0.3)
    fig_b.tight_layout()
    fig_b
    return


@app.cell
def _(TRUE_BIAS, TRUE_RHO, df_optima_b):
    bias_std = df_optima_b["best_bias"].std()
    rho_std = df_optima_b["best_rho"].std()
    bias_range = df_optima_b["best_bias"].max() - df_optima_b["best_bias"].min()
    rho_range = df_optima_b["best_rho"].max() - df_optima_b["best_rho"].min()
    pct_exact_bias = (df_optima_b["best_bias"] == TRUE_BIAS).mean() * 100
    pct_exact_rho = (df_optima_b["best_rho"] == TRUE_RHO).mean() * 100

    mo.md(
        f"""
        ### Statystyki rozkładu optymalnych parametrów

        | Parametr | Średnia | Odch. std. | Rozstęp | % sezonów = prawdziwa |
        |---|---|---|---|---|
        | bias | {df_optima_b['best_bias'].mean():.3f} | {bias_std:.3f} | {bias_range:.2f} | {pct_exact_bias:.0f}% |
        | ρ | {df_optima_b['best_rho'].mean():.3f} | {rho_std:.3f} | {rho_range:.2f} | {pct_exact_rho:.0f}% |

        Prawdziwe parametry: `bias = {TRUE_BIAS}`, `ρ = {TRUE_RHO}`.
        """
    )
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Interpretacja Eksperymentu B

    Wykres punktowy pokazuje, że **optymalne parametry skaczą od sezonu do
    sezonu**, mimo że dane generowane są z identycznego procesu. To nie jest
    sygnał, że prawdziwe parametry się zmieniają — to jest szum metryk.

    Przyczyna jest fundamentalna: metryka punktowa jest **funkcją dyskretną**
    (3/2/1/0) obliczaną na **jednej konkretnej predykcji** (argmax expected
    points). Na małej próbce (300 meczów) wystarczy kilka losowych zbieżności
    między nietypowym typem a rzeczywistym wynikiem, żeby przesunąć optimum
    o kilka kroków siatki.

    W przeciwieństwie do NLL, która agreguje informację ze **wszystkich**
    scenariuszy (każda komórka macierzy wpływa na wynik), punkty konkursowe
    zależą tylko od jednego wybranego typu. To sprawia, że ich powierzchnia
    jest pofałdowana i wrażliwa na konkretną realizację wyników.

    **Wniosek praktyczny:** optymalizacja parametrów modelu bezpośrednio pod
    punkty konkursowe wymaga próbek o jeden rząd wielkości większych niż
    typowy sezon ligowy (300 meczów to za mało). W praktyce lepiej dobrać
    parametry minimalizując NLL (stabilne optimum) i traktować metrykę
    punktową wyłącznie jako miarę walidacyjną na osobnym zbiorze.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Eksperyment C: Misspecyfikacja Modelu — Asymetryczny Bias

    Generujemy **3 000 meczów** „Faworyt vs Outsider":

    - **Lambdy bukmacherskie:** $\lambda_{\text{fav}}^{\text{bookie}} \sim U(1.5, 2.5)$,
      $\lambda_{\text{und}}^{\text{bookie}} \sim U(0.8, 1.3)$.
    - **Prawdziwe korekty** (asymetryczne):
      $\lambda_{\text{fav}}^{\text{true}} = \lambda_{\text{fav}}^{\text{bookie}} \cdot 0.85$,
      $\lambda_{\text{und}}^{\text{true}} = \lambda_{\text{und}}^{\text{bookie}} \cdot 1.15$.
    - Prawdziwe $\rho = -0.15$.

    Model może dopasować tylko **jeden globalny** `bias_correction` do obu
    lambd jednocześnie, co jest celową misspecyfikacją.
    """)
    return


@app.cell
def _(MAX_GOALS, SEED, TRUE_RHO):
    rng_c = np.random.default_rng(SEED + 2_000)
    n_c = 3_000

    bookie_home_c = rng_c.uniform(1.5, 2.5, size=n_c)
    bookie_away_c = rng_c.uniform(0.8, 1.3, size=n_c)

    true_home_c = bookie_home_c * 0.85
    true_away_c = bookie_away_c * 1.15

    builder_c = PoissonMatrixBuilder(rho=TRUE_RHO, max_goals_matrix=MAX_GOALS)
    home_scores_c = np.empty(n_c, dtype=int)
    away_scores_c = np.empty(n_c, dtype=int)
    for i in range(n_c):
        mat = builder_c.build_matrix(float(true_home_c[i]), float(true_away_c[i]))
        flat = rng_c.choice(mat.size, p=mat.ravel())
        home_scores_c[i], away_scores_c[i] = np.unravel_index(flat, mat.shape)

    data_c = {
        "base_home": bookie_home_c,
        "base_away": bookie_away_c,
        "true_home": true_home_c,
        "true_away": true_away_c,
        "home_score": home_scores_c,
        "away_score": away_scores_c,
    }

    mo.md(
        f"**Eksperyment C:** wygenerowano **{n_c}** meczów fav vs und. "
        f"Średnie gole: {home_scores_c.mean():.2f} (fav) / "
        f"{away_scores_c.mean():.2f} (und)."
    )
    return (data_c,)


@app.cell
def _(MAX_GOALS, compute_nll_grid, compute_pts_grid, data_c):
    bias_grid_c = np.round(np.arange(0.70, 1.301, 0.05), 3)
    rho_grid_c = np.round(np.arange(-0.30, 0.201, 0.05), 3)

    nll_c = compute_nll_grid(
        data_c["base_home"], data_c["base_away"],
        data_c["home_score"], data_c["away_score"],
        bias_grid_c, rho_grid_c, MAX_GOALS,
    )
    pts_c = compute_pts_grid(
        data_c["base_home"], data_c["base_away"],
        data_c["home_score"], data_c["away_score"],
        bias_grid_c, rho_grid_c, MAX_GOALS,
    )
    return bias_grid_c, nll_c, pts_c, rho_grid_c


@app.cell
def _(bias_grid_c, nll_c, pts_c, rho_grid_c):
    fig_c, axes_c = plt.subplots(1, 2, figsize=(14, 5.5))

    plot_metric_heatmap(
        axes_c[0], nll_c, bias_grid_c, rho_grid_c,
        "Standard NLL (niższe = lepsze)",
        true_bias=1.0, true_rho=-0.15, minimize=True,
    )
    plot_metric_heatmap(
        axes_c[1], pts_c, bias_grid_c, rho_grid_c,
        "Średnie punkty (wyższe = lepsze)",
        true_bias=1.0, true_rho=-0.15, minimize=False,
    )

    fig_c.suptitle(
        "Eksperyment C: NLL vs Punkty przy asymetrycznym biasie "
        "(prawdziwe korekty: fav×0.85, und×1.15)",
        fontsize=12, y=1.02,
    )
    fig_c.tight_layout()
    fig_c
    return


@app.cell
def _(bias_grid_c, nll_c, pts_c, rho_grid_c):
    def _best_c(grid, minimize):
        fn = np.nanargmin if minimize else np.nanargmax
        idx = np.unravel_index(int(fn(grid)), grid.shape)
        return float(bias_grid_c[idx[0]]), float(rho_grid_c[idx[1]]), float(grid[idx])

    nll_cb, nll_cr, nll_cv = _best_c(nll_c, True)
    pts_cb, pts_cr, pts_cv = _best_c(pts_c, False)

    summary_c = pd.DataFrame([
        {"Metryka": "Standard NLL", "bias*": nll_cb, "ρ*": nll_cr,
         "Wartość": f"{nll_cv:.4f}",
         "Interpretacja": "Kompromis — bliżej referencji kalibracyjnej"},
        {"Metryka": "Średnie punkty", "bias*": pts_cb, "ρ*": pts_cr,
         "Wartość": f"{pts_cv:.4f}",
         "Interpretacja": "Też kompromis, lecz dalszy od `bias`/`ρ` referencyjnych"},
    ])
    mo.vstack([
        mo.md("### Podsumowanie Eksperymentu C"),
        mo.ui.table(summary_c),
    ])
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ### Interpretacja Eksperymentu C

    Przy asymetrycznej misspecyfikacji model z jednym globalnym `bias` **nie
    może** jednocześnie idealnie dopasować faworytów i outsiderów — więc
    sensowne optimum na siatce jest z definicji **kompromisem**. Na wykresach
    widać to po obu stronach: ani NLL, ani punkty nie „stają" dokładnie na
    gwiazdce referencyjnej (`bias = 1.0`, `ρ = −0.15`), lecz wypadają w jej
    **sąsiedztwie**, szukając najlepszego przybliżenia przy złym założeniu
    struktury modelu.

    Różnica nie polega więc na tym, że tylko jedna metryka „idzie na kompromis",
    a druga na radykalną eksploatację podgrupy — **obie** szukają środka; inna
    jest **jakość** tego środka względem tego, co powinno być, gdy znamy DGP.
    **Standard NLL** agreguje informację z całego rozkładu (logarytmiczna kara za
    złe prawdopodobieństwo realizowanego wyniku), więc nie może tak łatwo
    „przechylić się" wyłącznie na wąski zestaw typów pod faworytów; w praktyce
    daje optimum **bliższe** referencji kalibracyjnej — w uruchomieniu z
    notatnika typowo `bias*` około **0.95** (niewielkie odchylenie od 1.0) oraz
    `ρ*` nieco odsunięte od −0.15, lecz w **mniejszym** stopniu niż przy
    optymalizacji punktów.

    **Średnie punkty** nadal są funkcją **jednej** decyzji typerskiej na mecz, więc
    powierzchnia jest bardziej podatna na przechylenie w stronę konfiguracji,
    która na danej próbce lepiej „łapie" łatwiejsze do strzelenia scenariusze —
    stąd **ten sam** charakter kompromisu, ale ze **słabszym** trafieniem w punkt,
    który reprezentuje prawdziwą kalibrację (np. `bias*` bliżej **0.9** oraz
    wyraźniejsze przesunięcie `ρ*` względem −0.15).

    **Wniosek:** misspecyfikacja **spłaszcza** różnicę między metrykami w sensie
    „kompromis vs brak kompromisu"; zamiast tego porównujemy **odległość**
    optymów od tego, co uznajemy za poprawną kalibrację. NLL pozostaje tu
    bezpieczniejszym kryterium doboru parametrów niż bezpośrednia maksymalizacja
    punktów — nawet gdy obie metryki technicznie wybierają środek wyważenia, a
    nie skraj strategię. Analogiczna sytuacja może się pojawić na danych
    rzeczywistych przy jednym globalnym `bias_correction` dla meczów o różnej
    dynamice bramkowej.
    """)
    return


@app.cell(hide_code=True)
def _():
    mo.md(r"""
    ---

    ## Podsumowanie i Wnioski

    | Eksperyment | Kluczowy wynik |
    |---|---|
    | **A** | Standard NLL jest regułą właściwą i odtwarza prawdziwe parametry. Weighted NLL jest podatne na „hacking" — optymalizator deformuje rozkład zamiast go kalibrować. |
    | **B** | Optymalizacja punktów na próbce 300 meczów daje niestabilne optimum — różne realizacje tego samego procesu prowadzą do zupełnie różnych parametrów. |
    | **C** | Przy misspecyfikacji **obie** metryki wybierają kompromis na siatce; NLL zwykle pozostaje **bliżej** referencji kalibracyjnej (`bias`, `ρ`), a punkty — dalej, przez dyskretny charakter nagrody 3/2/1/0. |

    ### Rekomendacje praktyczne

    1. **Do doboru parametrów używaj Standard NLL** — jest stabilna, prawidłowa
       i nie wymaga dużych próbek.
    2. **Punkty konkursowe traktuj wyłącznie jako miarę walidacyjną** na osobnym,
       nieprzeszukiwanym zbiorze danych.
    3. **Nigdy nie optymalizuj Weighted NLL** — jest podatna na systematyczne
       obchodzenie przez optymalizator.
    4. **Pracuj nad redukcją misspecyfikacji** (np. osobne korekty dla faworytów
       i outsiderów), zamiast liczyć na to, że metryka punktowa „sama to naprawi".
    """)
    return


if __name__ == "__main__":
    app.run()
