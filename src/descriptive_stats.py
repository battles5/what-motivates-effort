"""
Descriptive Statistics and Treatment Comparisons
Replicates Tables 1-3 from DellaVigna & Pope (2018)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from scipy import stats
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import (
    DATA_SHORT, DATA_FULL, FIGURES_DIR, TABLES_DIR, IMAGES_DIR,
    TREATMENT_ORDER, TREATMENT_NAMES,
)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
PALETTE = sns.color_palette("Set2", 18)


def load_data():
    """Load and prepare main dataset."""
    df = pd.read_stata(DATA_SHORT)
    # Ensure treatment is string
    df["treatment"] = df["treatment"].astype(str).str.strip()
    return df


def load_full_data():
    """Load full individual-level dataset."""
    df = pd.read_stata(DATA_FULL)
    df["treatment"] = df["treatment"].astype(str).str.strip()
    return df


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 1 — Summary Statistics per Treatment
# ═══════════════════════════════════════════════════════════════════════════════
def table1_summary_statistics(df):
    """Replicate Table 1: summary statistics by treatment."""
    summary = df.groupby("treatment")["buttonpresses"].agg(
        ["count", "mean", "std", "min", "median", "max"]
    )
    summary = summary.rename(columns={
        "count": "N", "mean": "Mean", "std": "Std Dev",
        "min": "Min", "median": "Median", "max": "Max",
    })
    summary["SE"] = summary["Std Dev"] / np.sqrt(summary["N"])

    # Reorder
    summary = summary.reindex([t for t in TREATMENT_ORDER if t in summary.index])
    summary.index = summary.index.map(lambda x: TREATMENT_NAMES.get(x, x))

    summary.to_csv(os.path.join(TABLES_DIR, "table1_summary_stats.csv"), float_format="%.2f")
    print("\n=== TABLE 1: Summary Statistics ===")
    print(summary.to_string(float_format=lambda x: f"{x:.2f}"))
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 2 — Pairwise Treatment Comparisons (t-tests)
# ═══════════════════════════════════════════════════════════════════════════════
def table2_treatment_comparisons(df):
    """Replicate Table 2: pairwise treatment effect comparisons vs baseline."""
    baseline = "1.3"  # No Payment
    bp_baseline = df.loc[df["treatment"] == baseline, "buttonpresses"]

    results = []
    for t in TREATMENT_ORDER:
        if t == baseline:
            continue
        bp_t = df.loc[df["treatment"] == t, "buttonpresses"]
        diff = bp_t.mean() - bp_baseline.mean()
        t_stat, p_val = stats.ttest_ind(bp_t, bp_baseline, equal_var=False)
        results.append({
            "Treatment": TREATMENT_NAMES.get(t, t),
            "Mean": bp_t.mean(),
            "Baseline Mean": bp_baseline.mean(),
            "Difference": diff,
            "t-stat": t_stat,
            "p-value": p_val,
            "Significant (5%)": "***" if p_val < 0.01 else ("**" if p_val < 0.05 else ("*" if p_val < 0.1 else "")),
        })

    tab2 = pd.DataFrame(results)
    tab2.to_csv(os.path.join(TABLES_DIR, "table2_treatment_comparisons.csv"), index=False, float_format="%.4f")
    print("\n=== TABLE 2: Treatment Comparisons vs No Payment ===")
    print(tab2.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return tab2


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 3 — OLS Regressions
# ═══════════════════════════════════════════════════════════════════════════════
def table3_ols_regressions(df):
    """Replicate Table 3: OLS treatment effects."""
    import statsmodels.api as sm

    # Create treatment dummies (omit "1.3" = No Payment as baseline)
    dummies = pd.get_dummies(df["treatment"], prefix="T", drop_first=False).astype(float)
    baseline_col = "T_1.3"
    X_cols = [c for c in dummies.columns if c != baseline_col]
    X = sm.add_constant(dummies[X_cols])
    y = df["buttonpresses"]

    model = sm.OLS(y, X).fit(cov_type="HC1")  # robust SEs as in the paper

    results = []
    for col in X_cols:
        t_name = col.replace("T_", "")
        results.append({
            "Treatment": TREATMENT_NAMES.get(t_name, t_name),
            "Coefficient": model.params[col],
            "Robust SE": model.bse[col],
            "t-stat": model.tvalues[col],
            "p-value": model.pvalues[col],
        })

    tab3 = pd.DataFrame(results)
    tab3.to_csv(os.path.join(TABLES_DIR, "table3_ols_regressions.csv"), index=False, float_format="%.4f")

    # Also save the intercept (= baseline mean)
    print("\n=== TABLE 3: OLS Treatment Effects (robust SEs) ===")
    print(f"Constant (No Payment mean): {model.params['const']:.2f} (SE: {model.bse['const']:.2f})")
    print(tab3.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Save full model summary
    with open(os.path.join(TABLES_DIR, "table3_ols_full_summary.txt"), "w") as f:
        f.write(model.summary().as_text())

    return tab3, model


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 1 — Bar Chart of Mean Effort by Treatment
# ═══════════════════════════════════════════════════════════════════════════════
def figure1_treatment_means(df):
    """Replicate Figure 2 from the paper: mean effort by treatment with CIs."""
    means = df.groupby("treatment")["buttonpresses"].agg(["mean", "sem"])
    means = means.reindex([t for t in TREATMENT_ORDER if t in means.index])
    means.index = means.index.map(lambda x: TREATMENT_NAMES.get(x, x))

    fig, ax = plt.subplots(figsize=(14, 6))
    colors = []
    categories = {
        "Piece Rate": [0, 1, 2, 3, 4],
        "Charity": [5, 6],
        "Time": [7, 8],
        "Gain/Loss": [9, 10, 11],
        "Probability": [12, 13],
        "Psychology": [14, 15, 16, 17],
    }
    cat_colors = {
        "Piece Rate": "#4C72B0",
        "Charity": "#55A868",
        "Time": "#C44E52",
        "Gain/Loss": "#8172B2",
        "Probability": "#CCB974",
        "Psychology": "#64B5CD",
    }
    bar_colors = ["gray"] * len(means)
    for cat, idxs in categories.items():
        for i in idxs:
            if i < len(bar_colors):
                bar_colors[i] = cat_colors[cat]

    bars = ax.bar(
        range(len(means)), means["mean"], yerr=1.96 * means["sem"],
        color=bar_colors, edgecolor="white", capsize=3, alpha=0.85,
    )
    ax.set_xticks(range(len(means)))
    ax.set_xticklabels(means.index, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Mean Button Presses", fontsize=12)
    ax.set_title("Average Effort by Treatment (with 95% CI)", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    # Add horizontal line at no-payment mean
    no_pay_mean = df.loc[df["treatment"] == "1.3", "buttonpresses"].mean()
    ax.axhline(y=no_pay_mean, color="red", linestyle="--", alpha=0.6, label=f"No Payment = {no_pay_mean:.0f}")
    ax.legend(fontsize=10)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig1_treatment_means.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig1_treatment_means.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved figure: fig1_treatment_means.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 2 — Distribution of Button Presses
# ═══════════════════════════════════════════════════════════════════════════════
def figure2_effort_distribution(df):
    """Histogram of effort distribution overall and by key treatments."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharey=False)

    # Overall
    axes[0].hist(df["buttonpresses"], bins=50, color="#4C72B0", edgecolor="white", alpha=0.8)
    axes[0].set_title("All Treatments", fontweight="bold")
    axes[0].set_xlabel("Button Presses")
    axes[0].set_ylabel("Frequency")

    # By piece rate level
    for t, label, color in [("1.3", "No Payment", "#C44E52"), ("1.1", "1c Piece Rate", "#55A868"), ("1.2", "10c Piece Rate", "#4C72B0")]:
        subset = df.loc[df["treatment"] == t, "buttonpresses"]
        axes[1].hist(subset, bins=30, alpha=0.5, label=label, edgecolor="white", color=color)
    axes[1].set_title("Piece Rate Comparison", fontweight="bold")
    axes[1].set_xlabel("Button Presses")
    axes[1].legend(fontsize=8)

    # Behavioral treatments
    for t, label, color in [("5.1", "Gain 40c", "#55A868"), ("5.2", "Loss 40c", "#C44E52"), ("10", "Gift Exchange", "#8172B2")]:
        subset = df.loc[df["treatment"] == t, "buttonpresses"]
        axes[2].hist(subset, bins=30, alpha=0.5, label=label, edgecolor="white", color=color)
    axes[2].set_title("Behavioral Treatments", fontweight="bold")
    axes[2].set_xlabel("Button Presses")
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig2_effort_distribution.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig2_effort_distribution.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure: fig2_effort_distribution.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURE 3 — Piece-Rate Response Curve
# ═══════════════════════════════════════════════════════════════════════════════
def figure3_piece_rate_curve(df):
    """Mean effort as a function of piece rate (treatments 1.3, 2, 1.1, 1.4, 1.2)."""
    piece_rates = {"1.3": 0, "2": 0.001, "1.1": 0.01, "1.4": 0.04, "1.2": 0.10}
    data = []
    for t, p in piece_rates.items():
        bp = df.loc[df["treatment"] == t, "buttonpresses"]
        data.append({"piece_rate": p, "mean": bp.mean(), "se": bp.sem(), "label": TREATMENT_NAMES[t]})
    data = pd.DataFrame(data).sort_values("piece_rate")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(data["piece_rate"], data["mean"], yerr=1.96*data["se"],
                fmt="o-", color="#4C72B0", capsize=5, markersize=8, linewidth=2)
    for _, row in data.iterrows():
        ax.annotate(row["label"], (row["piece_rate"], row["mean"]),
                    textcoords="offset points", xytext=(10, 10), fontsize=8)

    ax.set_xlabel("Piece Rate ($ per 100 presses)", fontsize=12)
    ax.set_ylabel("Mean Button Presses", fontsize=12)
    ax.set_title("Effort Response to Piece Rate", fontsize=14, fontweight="bold")
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("{x:,.0f}"))

    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "fig3_piece_rate_curve.pdf"), dpi=300, bbox_inches="tight")
    fig.savefig(os.path.join(IMAGES_DIR, "fig3_piece_rate_curve.pdf"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved figure: fig3_piece_rate_curve.pdf")


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(TABLES_DIR, exist_ok=True)
    os.makedirs(IMAGES_DIR, exist_ok=True)

    df = load_data()
    print(f"Loaded {len(df)} observations, {df['treatment'].nunique()} treatments\n")

    table1_summary_statistics(df)
    table2_treatment_comparisons(df)
    tab3, ols_model = table3_ols_regressions(df)
    figure1_treatment_means(df)
    figure2_effort_distribution(df)
    figure3_piece_rate_curve(df)

    print("\n✓ Descriptive statistics complete. Tables saved in output/tables/, figures in output/figures/")
