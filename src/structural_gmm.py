"""
Structural Estimation — Minimum Distance (GMM)
Replicates Table 5 (GMM columns) from DellaVigna & Pope (2018)
"""

import numpy as np
import pandas as pd
from sklearn.utils import resample
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_SHORT, TABLES_DIR

N_BOOTSTRAP = 2000
np.random.seed(42)


def load_and_prepare():
    dt = pd.read_stata(DATA_SHORT)
    dt["treatment"] = dt["treatment"].astype(str).str.strip()
    dt["buttonpresses_nearest_100"] = np.round(
        (dt["buttonpresses"] + 0.1) / 100
    ).clip(lower=0.25)
    return dt


def compute_empirical_moments(dt):
    """Compute treatment means (rounded to nearest 100)."""
    return dt.groupby("treatment")["buttonpresses_nearest_100"].mean()


def bootstrap_moments(dt, n_boot=N_BOOTSTRAP):
    """Bootstrap treatment means for SE computation."""
    treatments = ["1.1", "1.2", "1.3", "3.1", "3.2", "10", "4.1", "4.2"]
    boot_means = {t: [] for t in treatments}

    for _ in range(n_boot):
        for t in treatments:
            subset = dt.loc[dt["treatment"] == t, "buttonpresses_nearest_100"]
            boot_sample = resample(subset, replace=True)
            boot_means[t].append(np.round(boot_sample.mean(), 2))

    return {t: np.array(v) for t, v in boot_means.items()}


# ═══════════════════════════════════════════════════════════════════════════════
#  Minimum Distance — Closed-Form Solutions
# ═══════════════════════════════════════════════════════════════════════════════

def md_exponential(E11, E12, E13, E31, E32, E10, E41, E42):
    """
    Minimum distance estimation for exponential cost function.
    Uses 3 moments (treatments 1.1, 1.2, 1.3) to solve for k, γ, s.
    Then derives behavioral parameters from remaining moments.
    """
    P = np.array([0.0, 0.01, 0.1])  # piece rates for 1.3, 1.1, 1.2

    # Solve system of 3 equations for k, gamma, s
    log_k = (np.log(P[2]) - np.log(P[1]) * (E12 / E11)) / (1 - (E12 / E11))
    log_gamma = np.log((np.log(P[1]) - log_k) / E11)
    log_s = np.exp(log_gamma) * E13 + log_k

    k = np.exp(log_k)
    g = np.exp(log_gamma)
    s = np.exp(log_s)

    # Derive behavioral parameters
    EG31 = np.exp(E31 * g)
    EG32 = np.exp(E32 * g)
    EG10 = np.exp(E10 * g)
    EG41 = np.exp(E41 * g)
    EG42 = np.exp(E42 * g)

    alpha = (100 / 9) * k * (EG32 - EG31)
    a = 100 * k * EG31 - 100 * s - alpha
    s_ge = k * EG10 - s
    delta = np.sqrt((k * EG42 - s) / (k * EG41 - s))
    beta = 100 * (k * EG41 - s) / (delta ** 2)

    return k, g, s, alpha, a, s_ge, beta, delta


def md_power(E11, E12, E13, E31, E32, E10, E41, E42):
    """
    Minimum distance estimation for power cost function.
    """
    P = np.array([0.0, 0.01, 0.1])

    log_k = (np.log(P[2]) - np.log(P[1]) * np.log(E12) / np.log(E11)) / \
            (1 - np.log(E12) / np.log(E11))
    log_gamma = np.log((np.log(P[1]) - log_k) / np.log(E11))
    log_s = np.exp(log_gamma) * np.log(E13) + log_k

    k = np.exp(log_k)
    g = np.exp(log_gamma)
    s = np.exp(log_s)

    alpha = (100 / 9) * k * (E32 ** g - E31 ** g)
    a = 100 * k * E31 ** g - 100 * s - alpha
    s_ge = k * E10 ** g - s
    delta = np.sqrt((k * E42 ** g - s) / (k * E41 ** g - s))
    beta = 100 * (k * E41 ** g - s) / (delta ** 2)

    return k, g, s, alpha, a, s_ge, beta, delta


def estimate_md(dt):
    """Run minimum distance estimation with bootstrap SEs."""
    moments = compute_empirical_moments(dt)

    # Point estimates
    print("\n=== TABLE 5 PANEL A: Benchmark GMM/MD ===")
    for spec, func in [("Exponential", md_exponential), ("Power", md_power)]:
        try:
            k, g, s, alpha, a, s_ge, beta, delta = func(
                moments["1.1"], moments["1.2"], moments["1.3"],
                moments["3.1"], moments["3.2"], moments["10"],
                moments["4.1"], moments["4.2"],
            )
            print(f"\n  {spec}:")
            print(f"    k = {k:.6e}")
            print(f"    γ = {g:.6f}")
            print(f"    s = {s:.6e}")
            print(f"    α (altruism) = {alpha:.6f}")
            print(f"    a (warm glow) = {a:.6f}")
            print(f"    Δs_GE (gift) = {s_ge:.6f}")
            print(f"    β (present bias) = {beta:.6f}")
            print(f"    δ (discount) = {delta:.6f}")
        except Exception as e:
            print(f"  {spec} failed: {e}")

    # Bootstrap SEs
    print("\nRunning bootstrap (n=2000)...")
    boot = bootstrap_moments(dt, N_BOOTSTRAP)

    results = {}
    for spec, func in [("exp", md_exponential), ("pow", md_power)]:
        boot_ests = []
        for i in range(N_BOOTSTRAP):
            try:
                est = func(
                    boot["1.1"][i], boot["1.2"][i], boot["1.3"][i],
                    boot["3.1"][i], boot["3.2"][i], boot["10"][i],
                    boot["4.1"][i], boot["4.2"][i],
                )
                boot_ests.append(est)
            except Exception:
                continue

        boot_arr = np.array(boot_ests)
        labels = ["k", "gamma", "s", "alpha", "a", "s_ge", "beta", "delta"]

        point = func(
            moments["1.1"], moments["1.2"], moments["1.3"],
            moments["3.1"], moments["3.2"], moments["10"],
            moments["4.1"], moments["4.2"],
        )

        spec_results = {}
        print(f"\n  {spec.upper()} Bootstrap SEs (n_valid={len(boot_ests)}):")
        for j, lab in enumerate(labels):
            val = point[j]
            se = np.nanstd(boot_arr[:, j])
            spec_results[lab] = val
            spec_results[f"{lab}_se"] = se
            print(f"    {lab:12s} = {val:12.6f}  (SE: {se:.6f})")

        results[spec] = spec_results

    return results


def save_gmm_results(results):
    """Save GMM results to CSV."""
    rows = []
    for spec in ["exp", "pow"]:
        r = results.get(spec, {})
        rows.append({
            "Model": f"GMM ({spec})",
            "gamma": r.get("gamma", np.nan),
            "gamma_se": r.get("gamma_se", np.nan),
            "k": r.get("k", np.nan),
            "k_se": r.get("k_se", np.nan),
            "s": r.get("s", np.nan),
            "s_se": r.get("s_se", np.nan),
            "alpha": r.get("alpha", np.nan),
            "alpha_se": r.get("alpha_se", np.nan),
            "a": r.get("a", np.nan),
            "a_se": r.get("a_se", np.nan),
            "s_ge": r.get("s_ge", np.nan),
            "s_ge_se": r.get("s_ge_se", np.nan),
            "beta": r.get("beta", np.nan),
            "beta_se": r.get("beta_se", np.nan),
            "delta": r.get("delta", np.nan),
            "delta_se": r.get("delta_se", np.nan),
        })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(TABLES_DIR, "table5_gmm_results.csv"), index=False)
    print(f"\nSaved to {os.path.join(TABLES_DIR, 'table5_gmm_results.csv')}")


if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)
    dt = load_and_prepare()
    print(f"Loaded {len(dt)} observations")
    results = estimate_md(dt)
    save_gmm_results(results)
    print("\n✓ GMM/MD structural estimation complete.")
