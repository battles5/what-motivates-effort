# Replication and Extension of 'What Motivates Effort?'

**Repository Description:** Replication and algorithmic extension of the economic research paper 'What Motivates Effort?' by S. DellaVigna and D. Pope (2018). This project implements structural models, machine learning, and bayesian statistics in Python to analyze behavioral economics data.

---

## 1. Theoretical and Methodological Introduction

**Disclaimer:** This repository and the code within it have no pretense of academic or scientific value. It is strictly an educational and didactic project developed as the final coding assignment for the course *Experiments and Real-World Evidence in Economics* of the Second-level Interuniversity Master *Master in Data Science and Statistical Learning* (MD2SL), jointly offered by the University of Florence (Università degli Studi di Firenze) and the IMT School for Advanced Studies Lucca (Scuola IMT Alti Studi Lucca).

The fundamental goal of this project is to algorithmically replicate the empirical findings of the 2018 paper *”What Motivates Effort? Evidence and Expert Forecasts”* (Review of Economic Studies) and to expand its methodological horizon. While the original paper relies on standard econometric software to estimate average behavioral responses (e.g., altruism, present bias, social comparison) to different incentives, this repository translates the structural economic models into Python. By doing so, it introduces modern computational techniques to expose underlying heterogeneous patterns in the human effort distribution.

## 2. What the Code Does

The codebase takes the raw experimental data from a large-scale real-effort task distributed via Amazon Mechanical Turk and performs a complete analytical pipeline. Specifically, it:
- Cleans and prepares the dataset for structural estimation.
- Replicates the core summary statistics and treatment effects across 18 randomly assigned incentive conditions.
- Estimates the parameters of a theoretical cost-of-effort model (capturing intrinsic motivation, time discounting, and social preferences) by numerically solving systems of non-linear equations.
- Injects advanced statistical and computational methods to verify the robustness of those parameters and to test for varying responses across different psychological profiles in the population.

## 3. How It Works

The analysis relies on a programmatic sequence of mathematical optimization and statistical learning. First, it uses Non-Linear Least Squares and Minimum Distance estimators to invert first-order conditions derived from microeconomic theory, fitting theoretical curves to the empirical averages. 

Once the baseline replication is achieved, the code deploys extension routines:
- **Causal Machine Learning:** It applies double machine learning frameworks (like Causal Forests) to predict how single individuals, rather than the average population, respond to targeted incentives.
- **Probabilistic Programming:** It uses Markov Chain Monte Carlo sampling to extract full posterior distributions for the structural parameters, moving away from simple point estimates and standard errors to fully quantify statistical uncertainty.
- **Unsupervised Learning:** It fits Gaussian Mixture Models to categorize the continuous spectrum of worker effort into distinct latent classes (e.g., highly motivated vs. minimal effort profiles).

## 4. How It Is Structured

The repository is modular and organized into independent analytical steps:

* **src/**: Contains the core Python routines.
  * config.py: Centralizes directory paths and constants.
  * descriptive_stats.py: Generates the baseline demographic and treatment tables.
  * structural_nls.py: Performs the non-linear optimization for the core parameters.
  * structural_gmm.py: Implements the alternative minimum distance estimation.
  * extensions.py: Houses all the advanced computational additions (Causal Forests, Mixture Models, Bayesian inferences, Quantile Regressions).
  * un_all.py: The master execution script that runs the entire pipeline sequentially.
* **data/** *(Not included in the public repository for privacy and copyright reasons)*: The directory expected to hold the raw survey data.
* **output/**: The destination folder where the code saves the generated plots, graphs, and aggregated data tables.
