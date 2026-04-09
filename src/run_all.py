"""
Main Analysis Pipeline — DellaVigna & Pope (2018) Replication + Extensions
Run all analyses in sequence and generate all outputs.
"""

import os
import sys

# Ensure src/ is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, IMAGES_DIR

# Create output directories
for d in [OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)

print("=" * 70)
print("  DellaVigna & Pope (2018) — Replication & Extensions")
print("  'What Motivates Effort? Evidence and Expert Forecasts'")
print("=" * 70)

# ── Step 1: Descriptive Statistics ────────────────────────────────────────
print("\n\n" + "─" * 70)
print("  STEP 1: Descriptive Statistics & Treatment Comparisons")
print("─" * 70)
from descriptive_stats import (
    load_data, table1_summary_statistics, table2_treatment_comparisons,
    table3_ols_regressions, figure1_treatment_means, figure2_effort_distribution,
    figure3_piece_rate_curve,
)

df = load_data()
print(f"Loaded {len(df)} observations, {df['treatment'].nunique()} treatments")

table1_summary_statistics(df)
table2_treatment_comparisons(df)
table3_ols_regressions(df)
figure1_treatment_means(df)
figure2_effort_distribution(df)
figure3_piece_rate_curve(df)

# ── Step 2: Structural Estimation — NLS ──────────────────────────────────
print("\n\n" + "─" * 70)
print("  STEP 2: Structural Estimation — NLS (Table 5 & 6)")
print("─" * 70)
from structural_nls import (
    load_and_prepare as nls_prepare, estimate_benchmark,
    estimate_full, estimate_prob_weighting, save_results as save_nls,
)

dt = nls_prepare()
benchmark = estimate_benchmark(dt)
full = estimate_full(dt)
prob_wt = estimate_prob_weighting(dt)
save_nls(benchmark, full, prob_wt)

# ── Step 3: Structural Estimation — GMM/MD ───────────────────────────────
print("\n\n" + "─" * 70)
print("  STEP 3: Structural Estimation — GMM/Minimum Distance (Table 5)")
print("─" * 70)
from structural_gmm import (
    load_and_prepare as gmm_prepare, estimate_md, save_gmm_results,
)

dt_gmm = gmm_prepare()
gmm_results = estimate_md(dt_gmm)
save_gmm_results(gmm_results)

# ── Step 4: Extensions ───────────────────────────────────────────────────
print("\n\n" + "─" * 70)
print("  STEP 4: Extensions (Causal Forest, Random Forest, Quantile Reg)")
print("─" * 70)
from extensions import (
    load_data as load_ext_data, causal_forest_analysis,
    random_forest_importance, quantile_regression, model_comparison_plot,
)

df_short, df_full = load_ext_data()
causal_forest_analysis(df_full)
random_forest_importance(df_short)
quantile_regression(df_short)
model_comparison_plot(df_short)

# ── Done ─────────────────────────────────────────────────────────────────
print("\n\n" + "=" * 70)
print("  ALL ANALYSES COMPLETE")
print(f"  Tables: {TABLES_DIR}")
print(f"  Figures: {FIGURES_DIR}")
print("=" * 70)
