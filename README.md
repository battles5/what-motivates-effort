# What Motivates Effort? &mdash; Replication and Extensions

> **Full Python replication and methodological extension of [DellaVigna, S. & Pope, D. (2018). "What Motivates Effort? Evidence and Expert Forecasts." *Review of Economic Studies*, 85(2): 1029-1069](https://doi.org/10.1093/restud/rdx033).**

---

## Academic Disclaimer

This repository has **no pretense of original academic or scientific value**. It is a purely educational and didactic project, developed as the final assignment for the course *Experiments and Real-World Evidence in Economics* (Professors Leonardo Boncinelli and Ennio Bilancini), part of the **Second-Level Interuniversity Master in Data Science and Statistical Learning (MD2SL)**, jointly offered by:

- **University of Florence** (Universita degli Studi di Firenze)
- **IMT School for Advanced Studies Lucca** (Scuola IMT Alti Studi Lucca)

The data used in this project is the property of the original authors and is not redistributed in this repository.

---

## Table of Contents

1. [Theoretical Background](#1-theoretical-background)
2. [What This Code Does](#2-what-this-code-does)
3. [Repository Structure](#3-repository-structure)
4. [Module Reference](#4-module-reference)
5. [Installation](#5-installation)
6. [Usage](#6-usage)
7. [Key Results](#7-key-results)
8. [Original Extensions](#8-original-extensions)
9. [References](#9-references)
10. [Author](#10-author)

---

## 1. Theoretical Background

The original paper investigates a deceptively simple question: **what motivates people to exert effort?** To answer it, DellaVigna and Pope designed a large-scale experiment on Amazon Mechanical Turk (N = 9,861) in which participants were asked to press alternating keyboard keys for 10 minutes under 18 different incentive treatments.

The treatments span six theoretical channels:
| Channel | Treatments |
|---|---|
| **Piece-rate incentives** | No pay, 0.1c, 1c, 4c, 10c per 100 presses |
| **Charity (social preferences)** | 1c and 10c donated to the Red Cross |
| **Time discounting** | 1c paid with 2-week and 4-week delay |
| **Gain/Loss framing** | Gain 40c, Loss 40c, Gain 80c |
| **Probability weighting** | 1% chance of $1, 50% chance of 2c |
| **Psychology** | Social comparison, ranking, task significance, gift exchange |

The key analytical tool is a **structural cost-of-effort model** in which an agent maximizes:

$$\max_{e \geq 0} \; (s + p) \cdot e - c(e)$$

where $s$ captures intrinsic motivation, $p$ is the extrinsic piece rate, and $c(e)$ is a convex cost function (parameterized in both exponential and power form). The first-order conditions are inverted to estimate the behavioral parameters from the observed treatment means.

This repository **replicates the full empirical pipeline in Python** (the original was coded in Stata) and **extends it with six additional computational methods** not present in the paper.

---

## 2. What This Code Does

The codebase performs a complete analytical pipeline in four stages:

1. **Descriptive Replication** &mdash; Reproduces Tables 1-3 and Figures 1-3 from the paper: summary statistics, pairwise treatment comparisons (Welch t-tests), and OLS regressions with heteroskedasticity-robust standard errors.
2. **Structural Estimation** &mdash; Estimates the cost-of-effort parameters ($\gamma$, $k$, $s$, $\alpha$, $a$, $\beta$, $\delta$) via Non-Linear Least Squares (`scipy.optimize.curve_fit`) and closed-form Minimum Distance with 2,000-draw bootstrap standard errors.
3. **Machine Learning Extensions** &mdash; Heterogeneous treatment effect estimation via Causal Forests (Double Machine Learning), distributional analysis via Quantile Regressions, and feature importance via Random Forests.
4. **Advanced Extensions** &mdash; Full Bayesian posterior inference via Markov Chain Monte Carlo (PyMC/NUTS), multiple testing corrections (Bonferroni, Holm, Benjamini-Hochberg), expert forecast replication, and latent worker-type discovery via Gaussian Mixture Models.

---

## 3. Repository Structure

```
.
|-- README.md
|-- requirements.txt
|-- .gitignore
|
|-- src/                            # Core Python analysis code
|   |-- config.py                   #   Paths, constants, treatment metadata
|   |-- descriptive_stats.py        #   Tables 1-3, Figures 1-3
|   |-- structural_nls.py           #   NLS estimation (Tables 5-6)
|   |-- structural_gmm.py           #   Minimum Distance estimation (Table 5)
|   |-- extensions.py               #   Causal Forest, Quantile Reg, Random Forest
|   |-- extensions_advanced.py      #   Bayesian, expert forecasts, mixture model
|   |-- run_all.py                  #   Master pipeline script
|
|-- output/
|   |-- figures/                    # Generated PDF figures
|   |   |-- fig1_treatment_means.pdf
|   |   |-- fig2_effort_distribution.pdf
|   |   |-- fig3_piece_rate_curve.pdf
|   |   |-- fig_bayesian_posteriors.pdf
|   |   |-- fig_causal_forest_te.pdf
|   |   |-- fig_expert_forecasts.pdf
|   |   |-- fig_mixture_model.pdf
|   |   |-- fig_model_comparison.pdf
|   |   |-- fig_multiple_testing.pdf
|   |   |-- fig_quantile_regression.pdf
|   |   |-- fig_rf_importance.pdf
|   |-- tables/                     # Generated CSV/TXT tables
|       |-- table1_summary_stats.csv
|       |-- table2_treatment_comparisons.csv
|       |-- table3_ols_regressions.csv
|       |-- table3_ols_full_summary.txt
|       |-- table5_6_nls_results.csv
|       |-- table5_gmm_results.csv
|       |-- expert_forecasts.csv
|       |-- multiple_testing.csv
|
|-- data/                           # NOT INCLUDED (see disclaimer)
```

---

## 4. Module Reference

### `config.py`
Central configuration. Defines all directory paths (`DATA_DIR`, `OUTPUT_DIR`, `FIGURES_DIR`, `TABLES_DIR`), dataset file references, and the canonical treatment ordering and human-readable name mapping used across the pipeline.

### `descriptive_stats.py`
Replicates the paper's descriptive analysis:
- `table1_summary_statistics()` &mdash; N, mean, standard deviation, min, median, max, standard error per treatment.
- `table2_treatment_comparisons()` &mdash; Pairwise Welch t-tests of each treatment against the "No Payment" baseline.
- `table3_ols_regressions()` &mdash; OLS regression with treatment dummies and Huber-White (HC1) robust standard errors via `statsmodels`.
- `figure1` through `figure3` &mdash; Publication-quality bar charts, histograms, and piece-rate response curves.

### `structural_nls.py`
Replicates Tables 5 and 6 (NLS columns):
- **Benchmark model** (3 parameters: $\gamma$, $k$, $s$) fitted on piece-rate treatments only.
- **Full model** (8 parameters: adds $\alpha$, $a$, $\Delta s_{GE}$, $\beta$, $\delta$) fitted jointly on piece-rate, charity, gift exchange, and delayed payment treatments.
- **Probability weighting** (4 parameters: adds $\pi_w$) for lottery-based treatments.
- Both exponential and power cost specifications are estimated in parallel.

### `structural_gmm.py`
Replicates Table 5 (GMM/Minimum Distance columns):
- Analytically solves a system of 3 moment equations from the first-order conditions to obtain $k$, $\gamma$, $s$ in closed form.
- Derives behavioral parameters ($\alpha$, $a$, $\beta$, $\delta$) from additional treatment moments.
- Computes standard errors via 2,000-iteration bootstrap resampling.

### `extensions.py`
Machine learning extensions not present in the paper:
- `causal_forest_analysis()` &mdash; Estimates individualized treatment effects using `econml.CausalForestDML` (Double Machine Learning). Computes ATE, confidence intervals, and the full distribution of heterogeneous effects.
- `random_forest_importance()` &mdash; Ranks treatments by predictive power using a Random Forest with 5-fold cross-validation.
- `quantile_regression()` &mdash; Estimates treatment effects at the 25th, 50th, and 75th percentiles via `statsmodels.QuantReg`.
- `model_comparison_plot()` &mdash; Compares structural model predictions against actual treatment means.

### `extensions_advanced.py`
Advanced statistical extensions:
- `bayesian_structural_estimation()` &mdash; Full posterior inference on the exponential cost function via PyMC (NUTS sampler, 2 chains, 2000 draws). Reports HDI intervals, convergence diagnostics, and WAIC.
- `expert_forecast_analysis()` &mdash; Replicates the expert prediction exercise: compares 208 academics' forecasts against actual outcomes, computing correlations, MAE, and systematic biases.
- `multiple_testing_corrections()` &mdash; Applies Bonferroni, Holm, and Benjamini-Hochberg corrections to the 17 treatment p-values.
- `finite_mixture_model()` &mdash; Fits Gaussian Mixture Models (K = 2 to 5) with BIC-based selection to identify latent worker behavioral types.

### `run_all.py`
Orchestration script. Executes the full pipeline sequentially: descriptive statistics, NLS, GMM, and the basic ML extensions. The advanced extensions (`extensions_advanced.py`) require separate execution due to the heavier computational requirements of MCMC sampling.

---

## 5. Installation

```bash
# Clone the repository
git clone https://github.com/battles5/what-motivates-effort.git
cd what-motivates-effort

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate        # Linux/macOS
.venv\Scripts\Activate.ps1       # Windows PowerShell

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

| Package | Purpose |
|---|---|
| `numpy`, `pandas` | Data manipulation and numerical computation |
| `scipy` | Non-linear optimization (`curve_fit`) and statistical tests |
| `matplotlib`, `seaborn` | Figure generation |
| `statsmodels` | OLS regressions, quantile regressions, multiple testing |
| `scikit-learn` | Random Forest, Gradient Boosting, Gaussian Mixture Models |
| `econml` | Causal Forest with Double Machine Learning |
| `pymc`, `arviz` | Bayesian MCMC estimation and diagnostics |

> **Note:** `econml` and `pymc` are optional. If not installed, the corresponding analyses are skipped gracefully.

---

## 6. Usage

### Run the main pipeline

```bash
cd src
python run_all.py
```

This executes the descriptive replication, structural estimation (NLS and Minimum Distance), and the ML-based extensions. Results are saved to `output/figures/` and `output/tables/`.

### Run the advanced extensions separately

```python
from extensions_advanced import *

df = load_data()
bayesian_structural_estimation(df)    # MCMC (~2 min)
expert_forecast_analysis()            # Expert predictions
multiple_testing_corrections(df)      # p-value corrections
finite_mixture_model(df)              # Latent types
```

> **Data requirement:** The pipeline expects the original Stata datasets in `data/dellavigna_pope_2018/`. These are not distributed with this repository. They can be obtained from the [original paper's replication files](https://eml.berkeley.edu/~sdellavi/wp/MotivatesEffortQJE2018.zip).

---

## 7. Key Results

The Python replication closely matches the original Stata estimates:

| Parameter | Paper (Stata) | This Replication (Python) |
|---|---|---|
| Piece-rate response | Concave | Concave |
| Warm glow ($a$) | ~0.13 | 0.126 (NLS) / 0.143 (MD) |
| Altruism ($\alpha$) | ~0.004 | 0.004-0.005 |
| Present bias ($\beta$) | ~1 (no bias) | 1.00 (NLS) / 1.24 (MD) |
| $\gamma$ (exponential) | ~1.5 | 1.55 (NLS) / 1.56 (MD) / 1.61 (Bayes) |
| $\gamma$ (power) | ~33 | 33.0 (NLS) / 32.8 (MD) |
| Expert forecast correlation | 0.77 | 0.77 |
| Significant treatments (after Bonferroni) | 17/18 | 16/17 |

---

## 8. Original Extensions

Beyond the paper's analysis, this project contributes six computational extensions:

1. **Causal Forest (Double ML):** Reveals that treatment effects are heterogeneous ($\sigma = 144$ button presses), identifying *who* responds most to incentives rather than just the average effect.
2. **Quantile Regression:** Shows that behavioral treatments (especially loss framing) have disproportionately large effects on low-effort workers (+993 at $q = 0.25$ vs. +389 at $q = 0.75$).
3. **Bayesian Structural Estimation:** Produces full posterior uncertainty quantification for the structural parameters. $\gamma_{\text{Bayes}} = 1.61$ with 95% HDI [1.29, 2.01], consistent with NLS and MD point estimates.
4. **Multiple Testing Corrections:** 16 of 17 treatments remain significant after Bonferroni, Holm, and Benjamini-Hochberg corrections. Only Gift Exchange loses significance.
5. **Finite Mixture Model:** Identifies 5 latent worker types via BIC-selected Gaussian Mixture: from "shirkers" (7%, mean ~430 presses) to "motivated" (21%, mean ~2,727 presses).
6. **Random Forest Feature Importance:** Confirms that treatment assignment alone has limited individual-level predictive power, consistent with the large within-treatment heterogeneity.

---

## 9. References

**Original paper:**
- DellaVigna, S. & Pope, D. (2018). What Motivates Effort? Evidence and Expert Forecasts. *Review of Economic Studies*, 85(2): 1029-1069. [DOI](https://doi.org/10.1093/restud/rdx033)

**Reference code:**
- Pozzi, A. & Nunnari, S. (Bocconi University). Python replication notebooks.

**Key methodological references:**
- Athey, S. & Imbens, G. (2019). Machine Learning Methods That Economists Should Know About. *Annual Review of Economics*.
- Chernozhukov, V. et al. (2018). Double/Debiased Machine Learning. *Econometrics Journal*.
- Salvatier, J., Wiecki, T. V., & Fonnesbeck, C. (2016). Probabilistic programming in Python using PyMC3. *PeerJ Computer Science*.

---

## 10. Author

**Orso Peruzzi**
Course: *Experiments and Real-World Evidence in Economics*
Professors: Leonardo Boncinelli and Ennio Bilancini
Master MD2SL &mdash; University of Florence and IMT School for Advanced Studies Lucca

---

*This repository is provided as-is for educational purposes only.*