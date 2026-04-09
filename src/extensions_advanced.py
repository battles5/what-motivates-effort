"""
Advanced Extensions — Original Contributions Beyond the Paper
1. Bayesian Structural Estimation (PyMC)
2. Expert Forecast Replication & Analysis
3. Multiple Testing Corrections (Bonferroni, Holm, BH-FDR)
4. Finite Mixture Model (latent effort types)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_SHORT, DATA_FULL, DATA_EXPERTS,
    FIGURES_DIR, TABLES_DIR, IMAGES_DIR,
    TREATMENT_ORDER, TREATMENT_NAMES,
)

sns.set_theme(style="whitegrid", font_scale=1.1)


def load_data():
    df = pd.read_stata(DATA_SHORT)
    df["treatment"] = df["treatment"].astype(str).str.strip()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  1. BAYESIAN STRUCTURAL ESTIMATION (PyMC)
# ═══════════════════════════════════════════════════════════════════════════════

def bayesian_structural_estimation(df):
    """
    Bayesian estimation of the exponential cost function parameters
    using PyMC MCMC. Compares posterior distributions with NLS point estimates.

    Model: e* = (1/γ) * [log(s + p) - log(k)]
    Priors: γ ~ HalfNormal, log(k) ~ Normal, log(s) ~ Normal
    Likelihood: y ~ Normal(e*, σ)
    """
    import pymc as pm
    import arviz as az

    # Use piece-rate treatments only, aggregate to treatment means for speed
    piece_rates = {"1.3": 0.0, "2": 0.001, "1.1": 0.01, "1.4": 0.04, "1.2": 0.10}
    sub = df[df["treatment"].isin(piece_rates.keys())].copy()
    sub["piece_rate"] = sub["treatment"].map(piece_rates)
    sub["bp100"] = np.round((sub["buttonpresses"] + 0.1) / 100).clip(lower=0.25)

    # Use treatment-level data for faster MCMC
    agg = sub.groupby("treatment").agg(
        mean_bp100=("bp100", "mean"),
        se_bp100=("bp100", "sem"),
        n=("bp100", "count"),
        piece_rate=("piece_rate", "first"),
    ).reset_index()

    p_obs = agg["piece_rate"].values
    y_obs = agg["mean_bp100"].values
    se_obs = agg["se_bp100"].values

    print("\n=== BAYESIAN STRUCTURAL ESTIMATION (Exponential) ===")
    print(f"  Using {len(agg)} treatment means, total N={agg['n'].sum()}")

    with pm.Model() as model:
        # Priors
        gamma = pm.HalfNormal("gamma", sigma=5.0)
        log_k = pm.Normal("log_k", mu=-35, sigma=10)
        log_s = pm.Normal("log_s", mu=-12, sigma=5)

        # Structural prediction: e* = (1/γ) * [log(s+p) - log(k)]
        s = pm.math.exp(log_s)
        k = pm.math.exp(log_k)
        mu = (1.0 / gamma) * (pm.math.log(s + p_obs) - pm.math.log(k))

        # Likelihood (treatment means, known SE)
        sigma = pm.HalfNormal("sigma", sigma=2.0)
        y = pm.Normal("y", mu=mu, sigma=sigma, observed=y_obs)

        # Sample with higher target_accept to reduce divergences
        trace = pm.sample(2000, tune=2000, cores=1, chains=2,
                          random_seed=42, progressbar=True,
                          return_inferencedata=True,
                          target_accept=0.95,
                          idata_kwargs={"log_likelihood": True})

    # Extract posteriors
    summary = az.summary(trace, var_names=["gamma", "log_k", "log_s", "sigma"],
                         hdi_prob=0.95)
    print("\n  Posterior summary:")
    print(summary.to_string())

    # Compute derived parameters
    gamma_post = trace.posterior["gamma"].values.flatten()
    logk_post = trace.posterior["log_k"].values.flatten()
    logs_post = trace.posterior["log_s"].values.flatten()
    k_post = np.exp(logk_post)
    s_post = np.exp(logs_post)

    print(f"\n  Derived posteriors:")
    print(f"    γ:  mean={np.mean(gamma_post):.4f}, 95% HDI=[{np.percentile(gamma_post, 2.5):.4f}, {np.percentile(gamma_post, 97.5):.4f}]")
    print(f"    k:  mean={np.mean(k_post):.4e}, median={np.median(k_post):.4e}")
    print(f"    s:  mean={np.mean(s_post):.4e}, median={np.median(s_post):.4e}")

    # WAIC for model comparison
    try:
        waic = az.waic(trace)
        print(f"\n  WAIC = {waic.elpd_waic:.2f} (SE: {waic.se:.2f})")
    except Exception as e:
        print(f"\n  WAIC computation skipped: {e}")

    # --- Plot: Posterior distributions ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].hist(gamma_post, bins=50, color="#4C72B0", edgecolor="white", alpha=0.8, density=True)
    axes[0].axvline(x=1.554, color="red", linestyle="--", linewidth=2, label="NLS = 1.554")
    axes[0].set_title(r"Posterior of $\gamma$", fontweight="bold")
    axes[0].set_xlabel(r"$\gamma$ (curvature)")
    axes[0].legend(fontsize=9)

    axes[1].hist(logk_post, bins=50, color="#55A868", edgecolor="white", alpha=0.8, density=True)
    axes[1].axvline(x=np.log(1.94e-16), color="red", linestyle="--", linewidth=2, label="NLS")
    axes[1].set_title(r"Posterior of $\log(k)$", fontweight="bold")
    axes[1].set_xlabel(r"$\log(k)$")
    axes[1].legend(fontsize=9)

    axes[2].hist(logs_post, bins=50, color="#C44E52", edgecolor="white", alpha=0.8, density=True)
    axes[2].axvline(x=np.log(3.65e-6), color="red", linestyle="--", linewidth=2, label="NLS")
    axes[2].set_title(r"Posterior of $\log(s)$", fontweight="bold")
    axes[2].set_xlabel(r"$\log(s)$")
    axes[2].legend(fontsize=9)

    plt.suptitle("Bayesian Structural Estimation — Exponential Cost Function", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_bayesian_posteriors.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig_bayesian_posteriors.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_bayesian_posteriors.pdf")

    return trace, summary


# ═══════════════════════════════════════════════════════════════════════════════
#  2. EXPERT FORECAST REPLICATION
# ═══════════════════════════════════════════════════════════════════════════════

def expert_forecast_analysis():
    """
    Replicate the Expert Forecast analysis from the paper.
    208 academics predicted outcomes for treatments 4-18 (given 1-3 as anchors).
    """
    df_exp = pd.read_stata(DATA_EXPERTS)

    # Treatment mapping: t4=4c Piece Rate, ..., t18=Gift Exchange
    treatment_map = {
        4: ("1.4", "4c Piece Rate"),
        5: ("2", "Very Low Pay"),
        6: ("3.1", "1c Red Cross"),
        7: ("3.2", "10c Red Cross"),
        8: ("4.1", "1c 2 Weeks"),
        9: ("4.2", "1c 4 Weeks"),
        10: ("5.1", "Gain 40c"),
        11: ("5.2", "Loss 40c"),
        12: ("5.3", "Gain 80c"),
        13: ("6.1", "Prob .01 $1"),
        14: ("6.2", "Prob .5 2c"),
        15: ("7", "Social Comp."),
        16: ("8", "Ranking"),
        17: ("9", "Task Signif."),
        18: ("10", "Gift Exch."),
    }

    results = []
    for t_num, (t_code, t_name) in treatment_map.items():
        forecast_col = f"treatment_t{t_num}"
        actual_col = f"treatment_t{t_num}_actual"

        if forecast_col not in df_exp.columns:
            continue

        forecasts = df_exp[forecast_col].dropna()
        actual = df_exp[actual_col].iloc[0]

        forecast_mean = forecasts.mean()
        forecast_median = forecasts.median()
        error = forecast_mean - actual
        abs_error = abs(error)
        pct_error = 100 * error / actual

        results.append({
            "Treatment": t_name,
            "Actual": actual,
            "Forecast Mean": forecast_mean,
            "Forecast Median": forecast_median,
            "Error": error,
            "Abs Error": abs_error,
            "% Error": pct_error,
        })

    tab = pd.DataFrame(results)
    tab.to_csv(os.path.join(TABLES_DIR, "expert_forecasts.csv"), index=False, float_format="%.1f")

    print("\n=== EXPERT FORECAST ANALYSIS ===")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\n  Mean absolute error: {tab['Abs Error'].mean():.1f} presses")
    print(f"  Mean % error:        {tab['% Error'].mean():.1f}%")
    print(f"  Correlation (actual vs forecast mean): {np.corrcoef(tab['Actual'], tab['Forecast Mean'])[0,1]:.3f}")

    # --- Plot: Actual vs Expert Forecasts ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Scatter: actual vs forecast
    ax = axes[0]
    ax.scatter(tab["Actual"], tab["Forecast Mean"], s=80, color="#4C72B0", edgecolor="white", zorder=3)
    lims = [min(tab["Actual"].min(), tab["Forecast Mean"].min()) - 50,
            max(tab["Actual"].max(), tab["Forecast Mean"].max()) + 50]
    ax.plot(lims, lims, "k--", alpha=0.5, label="Perfect forecast")
    for _, row in tab.iterrows():
        ax.annotate(row["Treatment"], (row["Actual"], row["Forecast Mean"]),
                    fontsize=7, textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("Actual Mean (button presses)", fontsize=11)
    ax.set_ylabel("Expert Forecast Mean", fontsize=11)
    ax.set_title("Expert Forecasts vs Actual Outcomes", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    # Bar: forecast errors
    ax2 = axes[1]
    colors = ["#C44E52" if e > 0 else "#55A868" for e in tab["Error"]]
    bars = ax2.barh(range(len(tab)), tab["Error"], color=colors, edgecolor="white", alpha=0.85)
    ax2.set_yticks(range(len(tab)))
    ax2.set_yticklabels(tab["Treatment"], fontsize=9)
    ax2.set_xlabel("Forecast Error (forecast - actual)", fontsize=11)
    ax2.set_title("Expert Forecast Errors by Treatment", fontweight="bold", fontsize=13)
    ax2.axvline(x=0, color="black", linewidth=0.8)
    ax2.invert_yaxis()

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_expert_forecasts.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig_expert_forecasts.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_expert_forecasts.pdf")

    # --- Plot: Distribution of forecasts for selected treatments ---
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    selected = [(18, "Gift Exch."), (10, "Gain 40c"), (5, "Very Low Pay")]
    for ax, (t_num, t_name) in zip(axes, selected):
        forecasts = df_exp[f"treatment_t{t_num}"].dropna()
        actual = df_exp[f"treatment_t{t_num}_actual"].iloc[0]
        ax.hist(forecasts, bins=25, color="#4C72B0", edgecolor="white", alpha=0.8, density=True)
        ax.axvline(x=actual, color="red", linestyle="--", linewidth=2, label=f"Actual = {actual:.0f}")
        ax.axvline(x=forecasts.mean(), color="orange", linestyle="-", linewidth=2, label=f"Forecast mean = {forecasts.mean():.0f}")
        ax.set_title(t_name, fontweight="bold")
        ax.set_xlabel("Button presses")
        ax.legend(fontsize=8)

    plt.suptitle("Distribution of Expert Forecasts (208 Academics)", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_expert_forecast_dist.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig_expert_forecast_dist.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_expert_forecast_dist.pdf")

    return tab


# ═══════════════════════════════════════════════════════════════════════════════
#  3. MULTIPLE TESTING CORRECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def multiple_testing_corrections(df):
    """
    Apply Bonferroni, Holm, and Benjamini-Hochberg FDR corrections
    to the 17 pairwise t-tests (each treatment vs No Payment).
    """
    from statsmodels.stats.multitest import multipletests

    baseline = "1.3"
    bp_baseline = df.loc[df["treatment"] == baseline, "buttonpresses"]

    treatments = [t for t in TREATMENT_ORDER if t != baseline]
    raw_pvals = []
    t_names = []

    for t in treatments:
        bp_t = df.loc[df["treatment"] == t, "buttonpresses"]
        _, p_val = stats.ttest_ind(bp_t, bp_baseline, equal_var=False)
        raw_pvals.append(p_val)
        t_names.append(TREATMENT_NAMES.get(t, t))

    raw_pvals = np.array(raw_pvals)

    # Corrections
    _, p_bonf, _, _ = multipletests(raw_pvals, method="bonferroni")
    _, p_holm, _, _ = multipletests(raw_pvals, method="holm")
    _, p_bh, _, _ = multipletests(raw_pvals, method="fdr_bh")

    tab = pd.DataFrame({
        "Treatment": t_names,
        "Raw p-value": raw_pvals,
        "Bonferroni": p_bonf,
        "Holm": p_holm,
        "BH-FDR": p_bh,
        "Sig (raw 5%)": ["*" if p < 0.05 else "" for p in raw_pvals],
        "Sig (Bonf 5%)": ["*" if p < 0.05 else "" for p in p_bonf],
        "Sig (BH 5%)": ["*" if p < 0.05 else "" for p in p_bh],
    })

    tab.to_csv(os.path.join(TABLES_DIR, "multiple_testing.csv"), index=False, float_format="%.6f")

    print("\n=== MULTIPLE TESTING CORRECTIONS ===")
    print(tab.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    n_raw = sum(raw_pvals < 0.05)
    n_bonf = sum(p_bonf < 0.05)
    n_bh = sum(p_bh < 0.05)
    print(f"\n  Significant at 5%: raw={n_raw}/17, Bonferroni={n_bonf}/17, BH-FDR={n_bh}/17")

    # --- Plot: p-value comparison ---
    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(t_names))
    width = 0.22

    ax.bar(x - width, -np.log10(raw_pvals), width, label="Raw", color="#4C72B0", alpha=0.85)
    ax.bar(x, -np.log10(p_bonf), width, label="Bonferroni", color="#C44E52", alpha=0.85)
    ax.bar(x + width, -np.log10(p_bh), width, label="BH-FDR", color="#55A868", alpha=0.85)

    ax.axhline(y=-np.log10(0.05), color="black", linestyle="--", alpha=0.6, label="α = 0.05")
    ax.set_xticks(x)
    ax.set_xticklabels(t_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(r"$-\log_{10}(p)$", fontsize=12)
    ax.set_title("Multiple Testing Corrections: Treatment Effects vs No Payment", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_multiple_testing.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig_multiple_testing.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_multiple_testing.pdf")

    return tab


# ═══════════════════════════════════════════════════════════════════════════════
#  4. FINITE MIXTURE MODEL (Gaussian Mixture)
# ═══════════════════════════════════════════════════════════════════════════════

def finite_mixture_model(df):
    """
    Fit a Gaussian Mixture Model to identify latent effort types.
    Use BIC to select optimal number of components (2-5).
    Analyze how treatment assignment varies across latent types.
    """
    from sklearn.mixture import GaussianMixture

    y = df["buttonpresses"].values.reshape(-1, 1)

    # Select optimal K via BIC
    bics = []
    models = []
    K_range = range(2, 6)
    for k in K_range:
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=5)
        gmm.fit(y)
        bics.append(gmm.bic(y))
        models.append(gmm)

    best_k = list(K_range)[np.argmin(bics)]
    best_gmm = models[np.argmin(bics)]

    print(f"\n=== FINITE MIXTURE MODEL ===")
    print(f"  BIC by K: {dict(zip(K_range, [f'{b:.0f}' for b in bics]))}")
    print(f"  Optimal K = {best_k}")

    # Assign types
    df = df.copy()
    df["type"] = best_gmm.predict(y)

    # Sort types by mean effort
    type_means = df.groupby("type")["buttonpresses"].mean().sort_values()
    type_map = {old: new for new, old in enumerate(type_means.index)}
    df["type"] = df["type"].map(type_map)

    print(f"\n  Latent types:")
    for t in range(best_k):
        sub = df[df["type"] == t]
        print(f"    Type {t}: N={len(sub)} ({100*len(sub)/len(df):.1f}%), "
              f"mean={sub['buttonpresses'].mean():.0f}, "
              f"std={sub['buttonpresses'].std():.0f}")

    # Treatment composition by type
    cross = pd.crosstab(df["type"], df["treatment"], normalize="columns")
    print(f"\n  Treatment composition by type (column %): see output table")

    # --- Plot 1: Mixture components ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    x_range = np.linspace(0, 4500, 500)
    colors = ["#4C72B0", "#C44E52", "#55A868", "#8172B2", "#CCB974"]

    ax.hist(df["buttonpresses"], bins=60, density=True, color="lightgray", edgecolor="white", alpha=0.6, label="Data")
    for k_i in range(best_k):
        mean = best_gmm.means_[type_map[k_i] if k_i in type_map.values() else k_i][0]
        # Use the original component ordering for GMM parameters
        pass

    # Simpler: plot by assigned type
    for t in range(best_k):
        sub = df[df["type"] == t]["buttonpresses"]
        ax.hist(sub, bins=40, density=True, alpha=0.4, color=colors[t],
                label=f"Type {t} (N={len(sub)}, μ={sub.mean():.0f})")

    ax.set_xlabel("Button Presses", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title(f"Gaussian Mixture Model (K={best_k})", fontweight="bold", fontsize=13)
    ax.legend(fontsize=8)

    # Plot 2: Treatment means by type
    ax2 = axes[1]
    # Focus on key treatments
    key_ts = ["1.3", "1.1", "1.2", "5.2", "10"]
    type_by_treat = []
    for t in key_ts:
        for typ in range(best_k):
            sub = df[(df["treatment"] == t) & (df["type"] == typ)]
            type_by_treat.append({
                "Treatment": TREATMENT_NAMES.get(t, t),
                "Type": f"Type {typ}",
                "Proportion": len(sub) / len(df[df["treatment"] == t]),
            })

    tbt = pd.DataFrame(type_by_treat)
    pivot = tbt.pivot(index="Treatment", columns="Type", values="Proportion")
    pivot = pivot.reindex([TREATMENT_NAMES[t] for t in key_ts])
    pivot.plot(kind="bar", stacked=True, ax=ax2, color=colors[:best_k], edgecolor="white")
    ax2.set_ylabel("Proportion", fontsize=11)
    ax2.set_title("Type Composition by Treatment", fontweight="bold", fontsize=13)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=30, ha="right")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_mixture_model.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig_mixture_model.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_mixture_model.pdf")

    # --- Plot 3: BIC curve ---
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(list(K_range), bics, "o-", color="#4C72B0", markersize=8, linewidth=2)
    ax.scatter([best_k], [min(bics)], color="red", s=120, zorder=5, label=f"Best K={best_k}")
    ax.set_xlabel("Number of Components (K)", fontsize=12)
    ax.set_ylabel("BIC", fontsize=12)
    ax.set_title("Model Selection: BIC vs Number of Mixture Components", fontweight="bold", fontsize=13)
    ax.legend(fontsize=10)
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig_bic_selection.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig_bic_selection.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: fig_bic_selection.pdf")

    return best_gmm, df


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} observations\n")

    # 1. Bayesian
    print("=" * 60)
    print("  1. BAYESIAN STRUCTURAL ESTIMATION")
    print("=" * 60)
    trace, bayes_summary = bayesian_structural_estimation(df)

    # 2. Expert Forecasts
    print("\n" + "=" * 60)
    print("  2. EXPERT FORECAST ANALYSIS")
    print("=" * 60)
    expert_tab = expert_forecast_analysis()

    # 3. Multiple Testing
    print("\n" + "=" * 60)
    print("  3. MULTIPLE TESTING CORRECTIONS")
    print("=" * 60)
    mt_tab = multiple_testing_corrections(df)

    # 4. Mixture Model
    print("\n" + "=" * 60)
    print("  4. FINITE MIXTURE MODEL")
    print("=" * 60)
    gmm, df_typed = finite_mixture_model(df)

    print("\n\n✓ All advanced extensions complete.")
