"""
Extensions Beyond the Paper
- Causal Forest (econml) for heterogeneous treatment effects
- Random Forest feature importance
- Quantile regression
- Comparison plots
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import statsmodels.api as sm
import statsmodels.formula.api as smf
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_SHORT, DATA_FULL, FIGURES_DIR, TABLES_DIR, IMAGES_DIR, TREATMENT_NAMES

sns.set_theme(style="whitegrid", font_scale=1.1)


def load_data():
    df_short = pd.read_stata(DATA_SHORT)
    df_short["treatment"] = df_short["treatment"].astype(str).str.strip()

    df_full = pd.read_stata(DATA_FULL)
    df_full["treatment"] = df_full["treatment"].astype(str).str.strip()
    return df_short, df_full


# ═══════════════════════════════════════════════════════════════════════════════
#  1. CAUSAL FOREST — Heterogeneous Treatment Effects
# ═══════════════════════════════════════════════════════════════════════════════

def causal_forest_analysis(df_full):
    """
    Estimate heterogeneous treatment effects using Causal Forest (econml).
    Compare 10c piece rate vs No Payment, using individual covariates.
    """
    try:
        from econml.dml import CausalForestDML
    except ImportError:
        print("  econml not installed, skipping Causal Forest")
        return None

    # Binary treatment: 10c piece rate (1.2) vs no payment (1.3)
    sub = df_full[df_full["treatment"].isin(["1.2", "1.3"])].copy()
    sub["T"] = (sub["treatment"] == "1.2").astype(int)

    # Covariates: use available individual-level features
    # duration_time_minutes, practicecount as covariates
    covariates = ["duration_time_minutes", "practicecount"]
    available_covs = [c for c in covariates if c in sub.columns and sub[c].notna().sum() > 100]

    if len(available_covs) < 1:
        print("  Not enough covariates for Causal Forest, using simulated approach")
        return None

    sub = sub.dropna(subset=available_covs + ["buttonpresses"])
    X = sub[available_covs].values
    Y = sub["buttonpresses"].values
    T = sub["T"].values.reshape(-1, 1)

    cf = CausalForestDML(
        model_y=GradientBoostingRegressor(n_estimators=100, max_depth=3),
        model_t=GradientBoostingRegressor(n_estimators=100, max_depth=3),
        n_estimators=200,
        random_state=42,
    )
    cf.fit(Y, T, X=X)
    te = cf.effect(X)
    ate = cf.ate(X)
    ate_inf = cf.ate_inference(X)

    print(f"\n=== CAUSAL FOREST: 10c Piece Rate vs No Payment ===")
    print(f"  ATE = {ate:.2f}")
    try:
        ci = ate_inf.conf_int()
        print(f"  95% CI = [{ci[0][0]:.2f}, {ci[0][1]:.2f}]")
    except (AttributeError, TypeError):
        try:
            ci_lo = ate_inf.conf_int_mean()[0]
            ci_hi = ate_inf.conf_int_mean()[1]
            print(f"  95% CI = [{ci_lo:.2f}, {ci_hi:.2f}]")
        except Exception:
            print(f"  95% CI: (unavailable with this econml version)")
    print(f"  Treatment effect heterogeneity (std): {np.std(te):.2f}")

    # Plot distribution of individual treatment effects
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(te, bins=40, color="#4C72B0", edgecolor="white", alpha=0.8)
    ax.axvline(x=ate, color="red", linestyle="--", linewidth=2, label=f"ATE = {ate:.0f}")
    ax.set_xlabel("Individual Treatment Effect (button presses)", fontsize=12)
    ax.set_ylabel("Frequency", fontsize=12)
    ax.set_title("Distribution of Heterogeneous Treatment Effects\n(Causal Forest: 10c vs No Payment)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_causal_forest_te.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig_causal_forest_te.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_causal_forest_te.pdf")

    return {"ate": ate, "te_std": np.std(te)}


# ═══════════════════════════════════════════════════════════════════════════════
#  2. RANDOM FOREST — Feature Importance for Predicting Effort
# ═══════════════════════════════════════════════════════════════════════════════

def random_forest_importance(df_short):
    """
    Use Random Forest to predict effort from treatment dummies.
    Shows which treatments matter most for predicting effort level.
    """
    dummies = pd.get_dummies(df_short["treatment"], prefix="T")
    X = dummies.values
    y = df_short["buttonpresses"].values

    rf = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    rf.fit(X, y)

    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="r2")

    importances = pd.Series(rf.feature_importances_, index=dummies.columns)
    importances = importances.sort_values(ascending=True)
    importances.index = importances.index.map(
        lambda x: TREATMENT_NAMES.get(x.replace("T_", ""), x)
    )

    print(f"\n=== RANDOM FOREST: Treatment Importance ===")
    print(f"  5-fold CV R² = {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    print(f"  Top-5 important treatments:")
    for name, val in importances.tail(5).items():
        print(f"    {name}: {val:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    importances.plot(kind="barh", color="#55A868", edgecolor="white", ax=ax)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"Random Forest: Treatment Importance for Predicting Effort\n(5-fold CV R² = {cv_scores.mean():.3f})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_rf_importance.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig_rf_importance.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_rf_importance.pdf")

    return importances, cv_scores


# ═══════════════════════════════════════════════════════════════════════════════
#  3. QUANTILE REGRESSION — Effects at Different Points of Distribution
# ═══════════════════════════════════════════════════════════════════════════════

def quantile_regression(df_short):
    """
    Quantile regression at q=0.25, 0.50, 0.75 to study treatment effects
    across the effort distribution. Compare OLS (mean) with quantile effects.
    """
    # Focus on key treatments vs no payment (1.3)
    key_treatments = ["1.1", "1.2", "5.2", "10"]
    sub = df_short[df_short["treatment"].isin(["1.3"] + key_treatments)].copy()
    for t in key_treatments:
        sub[f"T_{t}"] = (sub["treatment"] == t).astype(int)

    formula = "buttonpresses ~ " + " + ".join([f"Q('{f'T_{t}'}')" for t in key_treatments])
    # Use simpler formula construction
    sub_reg = sub[["buttonpresses"] + [f"T_{t}" for t in key_treatments]].copy()
    X = sm.add_constant(sub_reg[[f"T_{t}" for t in key_treatments]])
    y = sub_reg["buttonpresses"]

    results_q = {}
    quantiles = [0.25, 0.50, 0.75]

    print("\n=== QUANTILE REGRESSION: Treatment Effects at q=0.25, 0.50, 0.75 ===")
    for q in quantiles:
        mod = sm.QuantReg(y, X).fit(q=q)
        results_q[q] = mod
        print(f"\n  Quantile = {q}:")
        for t in key_treatments:
            col = f"T_{t}"
            print(f"    {TREATMENT_NAMES[t]:20s}: {mod.params[col]:8.1f} (SE: {mod.bse[col]:.1f}, p={mod.pvalues[col]:.3f})")

    # Plot: Compare coefficients across quantiles + OLS
    ols = sm.OLS(y, X).fit(cov_type="HC1")

    fig, ax = plt.subplots(figsize=(10, 6))
    x_pos = np.arange(len(key_treatments))
    width = 0.18

    for i, q in enumerate(quantiles):
        coefs = [results_q[q].params[f"T_{t}"] for t in key_treatments]
        ses = [results_q[q].bse[f"T_{t}"] for t in key_treatments]
        ax.bar(x_pos + i * width, coefs, width, yerr=[1.96 * s for s in ses],
               label=f"q={q}", capsize=3, alpha=0.8)

    ols_coefs = [ols.params[f"T_{t}"] for t in key_treatments]
    ols_ses = [ols.bse[f"T_{t}"] for t in key_treatments]
    ax.bar(x_pos + 3 * width, ols_coefs, width, yerr=[1.96 * s for s in ols_ses],
           label="OLS (mean)", color="#C44E52", capsize=3, alpha=0.8)

    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels([TREATMENT_NAMES[t] for t in key_treatments], fontsize=10)
    ax.set_ylabel("Treatment Effect (button presses)", fontsize=12)
    ax.set_title("Treatment Effects Across the Effort Distribution\n(Quantile Regression vs OLS)", fontsize=13, fontweight="bold")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend(fontsize=10)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_quantile_regression.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig_quantile_regression.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_quantile_regression.pdf")

    return results_q


# ═══════════════════════════════════════════════════════════════════════════════
#  4. COMPARISON: OLS vs Structural Model Predictions
# ═══════════════════════════════════════════════════════════════════════════════

def model_comparison_plot(df_short):
    """
    Plot actual means vs OLS-predicted vs structural model predictions.
    This provides a visual comparison of how well the structural model fits.
    """
    means = df_short.groupby("treatment")["buttonpresses"].mean()

    # Piece-rate treatments for structural comparison
    piece_rates = {"1.3": 0, "2": 0.001, "1.1": 0.01, "1.4": 0.04, "1.2": 0.10}
    pr_data = pd.DataFrame([
        {"treatment": t, "piece_rate": p, "actual": means[t]}
        for t, p in piece_rates.items()
    ]).sort_values("piece_rate")

    # Read NLS results if available
    nls_path = os.path.join(TABLES_DIR, "table5_6_nls_results.csv")
    if os.path.exists(nls_path):
        nls_df = pd.read_csv(nls_path)
        bench_exp = nls_df[nls_df["Model"] == "Benchmark (exp)"].iloc[0]
        g = bench_exp["gamma"]
        k = bench_exp["k"]
        s = bench_exp["s"]
        pr_data["predicted_exp"] = (1 / g) * (np.log(s + pr_data["piece_rate"]) - np.log(k)) * 100
    else:
        pr_data["predicted_exp"] = np.nan

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(pr_data["piece_rate"], pr_data["actual"], "o-", color="#4C72B0",
            markersize=8, linewidth=2, label="Actual means")
    if not pr_data["predicted_exp"].isna().all():
        ax.plot(pr_data["piece_rate"], pr_data["predicted_exp"], "s--", color="#C44E52",
                markersize=8, linewidth=2, label="Structural (exponential)")
    ax.set_xlabel("Piece Rate ($ per 100 presses)", fontsize=12)
    ax.set_ylabel("Mean Button Presses", fontsize=12)
    ax.set_title("Model Fit: Actual vs Structural Predictions", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_model_comparison.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig_model_comparison.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("\nSaved: fig_model_comparison.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    df_short, df_full = load_data()
    print(f"Loaded {len(df_short)} (short), {len(df_full)} (full) observations\n")

    # 1. Causal Forest
    cf_result = causal_forest_analysis(df_full)

    # 2. Random Forest importance
    rf_imp, rf_cv = random_forest_importance(df_short)

    # 3. Quantile regression
    qr_results = quantile_regression(df_short)

    # 4. Model comparison (run after structural_nls.py)
    model_comparison_plot(df_short)

    print("\n✓ Extensions complete.")
