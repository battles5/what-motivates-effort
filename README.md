# What motivates effort? Evidence and expert forecasts, a replication and extension

## Overview

This repository contains the Python code and analytical pipeline used to replicate and extend the empirical analysis in DellaVigna, S. and Pope, D. (2018), "What Motivates Effort? Evidence and Expert Forecasts", *Review of Economic Studies*, 85(2), 1029-1069. The study examines what drives people to exert effort through a large-scale Amazon Mechanical Turk experiment with 18 incentive treatments spanning monetary rewards, social preferences, time discounting, reference dependence, and psychological motivators.

The replication translates the original Stata pipeline into Python, reproducing the descriptive statistics, treatment comparisons, and structural cost-of-effort estimation (Non-Linear Least Squares and closed-form Minimum Distance with bootstrap standard errors). Beyond replication, the codebase extends the analysis with Causal Forests for heterogeneous treatment effects, quantile regressions, Bayesian posterior inference via Markov Chain Monte Carlo, multiple testing corrections, Gaussian Mixture Models for latent worker types, and Random Forest feature importance.

> Disclaimer: this work represents the final project for the MD2SL Master's course in Experiments and Real-World Evidence in Economics (University of Florence and IMT Lucca). There is no claim to originality or academic value; it is purely a didactic exercise within the context of the course.

## Repository structure and code functionality

The analysis is divided into six modular Python scripts found in the `src/` directory:

1. `config.py`
   Central configuration module. Defines all directory paths, dataset file references, and the canonical treatment ordering and human-readable name mapping used across the pipeline.

2. `descriptive_stats.py`
   Replicates Tables 1-3 and Figures 1-3 from the paper. Loads the cleaned MTurk data, computes summary statistics per treatment, runs pairwise Welch t-tests against the No Payment baseline, fits an OLS regression with treatment dummies and Huber-White robust standard errors, and generates publication-quality PDF figures (bar charts, histograms, piece-rate response curve).

3. `structural_nls.py`
   Replicates Tables 5 and 6 (NLS columns). Implements the structural cost-of-effort model in both exponential and power specifications. Estimates the benchmark model (3 parameters: curvature, cost level, intrinsic motivation) on piece-rate treatments, a full model (8 parameters: adds altruism, warm glow, gift exchange, present bias, weekly discounting) on all behavioral treatments, and a probability weighting extension for lottery-based treatments via `scipy.optimize.curve_fit`.

4. `structural_gmm.py`
   Replicates Table 5 (Minimum Distance columns). Analytically solves a system of moment equations from the first-order conditions to obtain closed-form parameter estimates, then derives behavioral parameters from additional treatment moments. Computes standard errors via 2,000-iteration bootstrap resampling.

5. `extensions.py`
   Machine learning extensions not present in the original paper. Estimates individualized treatment effects using `econml.CausalForestDML` (Double Machine Learning), ranks treatments by predictive power using Random Forest with 5-fold cross-validation, estimates distributional treatment effects via quantile regression at the 25th, 50th, and 75th percentiles, and compares structural model predictions against actual treatment means.

6. `extensions_advanced.py`
   Advanced statistical extensions. Performs full Bayesian posterior inference on the exponential cost function via PyMC (NUTS sampler, 2 chains, 2,000 draws, WAIC diagnostics), replicates the expert prediction exercise comparing 208 academics' forecasts against actual outcomes, applies Bonferroni, Holm, and Benjamini-Hochberg corrections to the 17 treatment p-values, and fits Gaussian Mixture Models with BIC-based model selection to identify latent worker behavioral types.

The master script `run_all.py` orchestrates the main pipeline sequentially (descriptive statistics, NLS, Minimum Distance, and basic ML extensions). The advanced extensions require separate execution due to the heavier computational requirements of MCMC sampling.

Generated outputs (PDF figures and CSV tables) are saved to `output/figures/` and `output/tables/`.

## Usage guide

1. Clone the repository and navigate into the folder:
   ```
   git clone https://github.com/battles5/what-motivates-effort.git
   cd what-motivates-effort
   ```

2. Install the required Python dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Prepare the data: the scripts expect the original Stata datasets in `data/dellavigna_pope_2018/`. Since `data/` is excluded from version control, you will need to obtain the replication files from the authors and place them in the expected directory structure before running the pipeline.

4. Run the main analytical pipeline:
   ```
   cd src
   python run_all.py
   ```

5. Run the advanced extensions separately (Bayesian estimation, expert forecasts, multiple testing, mixture model):
   ```python
   from extensions_advanced import *
   df = load_data()
   bayesian_structural_estimation(df)
   expert_forecast_analysis()
   multiple_testing_corrections(df)
   finite_mixture_model(df)
   ```

Note: `econml` and `pymc` are optional dependencies. If not installed, the corresponding analyses are skipped gracefully.

## Included references and data sources

- DellaVigna, S. and Pope, D. (2018). What Motivates Effort? Evidence and Expert Forecasts. *Review of Economic Studies*, 85(2), 1029-1069.
- Pozzi, A. and Nunnari, S. (Bocconi University). Python replication notebooks (reference code).
- Athey, S. and Imbens, G. (2019). Machine Learning Methods That Economists Should Know About. *Annual Review of Economics*.
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C., Newey, W., and Robins, J. (2018). Double/Debiased Machine Learning for Treatment and Structural Parameters. *The Econometrics Journal*, 21(1).
- Salvatier, J., Wiecki, T. V., and Fonnesbeck, C. (2016). Probabilistic Programming in Python using PyMC3. *PeerJ Computer Science*.
- Wager, S. and Athey, S. (2018). Estimation and Inference of Heterogeneous Treatment Effects using Random Forests. *Journal of the American Statistical Association*.