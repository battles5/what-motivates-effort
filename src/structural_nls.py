"""
Structural Estimation — Non-Linear Least Squares (NLS)
Replicates Table 5 (NLS columns) and Table 6 from DellaVigna & Pope (2018)
"""

import numpy as np
import pandas as pd
import scipy.optimize as opt
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from config import DATA_SHORT, TABLES_DIR, FIGURES_DIR, IMAGES_DIR

# ── Scalers (same as reference code) ─────────────────────────────────────────
K_SCALER_EXP = 1e+16
S_SCALER_EXP = 1e+6
K_SCALER_POW = 1e+57
S_SCALER_POW = 1e+6


def load_and_prepare():
    """Load data and create treatment variables for NLS estimation."""
    dt = pd.read_stata(DATA_SHORT)
    dt["treatment"] = dt["treatment"].astype(str).str.strip()

    # Piece rate per 100 presses
    payoff_map = {"1.1": 0.01, "1.2": 0.1, "1.3": 0, "1.4": 0.04, "2": 0.001}
    dt["payoff_per_100"] = dt["treatment"].map(payoff_map).fillna(0)

    # Charity treatments
    charity_map = {"3.1": 0.01, "3.2": 0.1}
    dt["payoff_charity_per_100"] = dt["treatment"].map(charity_map).fillna(0)
    dt["dummy_charity"] = (dt["treatment"].isin(["3.1", "3.2"])).astype(int)

    # Time discounting
    delay_map = {"4.1": 2, "4.2": 4}
    dt["delay_wks"] = dt["treatment"].map(delay_map).fillna(0)
    dt["delay_dummy"] = (dt["treatment"].isin(["4.1", "4.2"])).astype(int)

    # Probability weighting
    prob_map = {"6.1": 0.01, "6.2": 0.5}
    dt["prob"] = dt["treatment"].map(prob_map).fillna(0)
    dt["weight_dummy"] = (dt["treatment"].isin(["6.1", "6.2"])).astype(int)

    # Gift exchange
    dt["gift_dummy"] = (dt["treatment"] == "10").astype(int)

    # Benchmark sample: piece-rate treatments only
    dt["dummy1"] = dt["treatment"].isin(["1.1", "1.2", "1.3", "1.4", "2"]).astype(int)

    # Round button presses to nearest 100 (with +0.1 for Python banker's rounding)
    dt["buttonpresses_nearest_100"] = np.round(
        (dt["buttonpresses"] + 0.1) / 100
    ).clip(lower=0.25)  # minimum 0.25 (= 25 presses)
    dt["logbuttonpresses_nearest_100"] = np.log(dt["buttonpresses_nearest_100"])

    return dt


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 5 PANEL A — Benchmark (Piece-Rate Only)
# ═══════════════════════════════════════════════════════════════════════════════

# --- Exponential cost ---
def benchmark_exp(pay100, g, k, s):
    k_val = k / K_SCALER_EXP
    s_val = s / S_SCALER_EXP + pay100
    return (1 / g) * (np.log(s_val) - np.log(k_val))


# --- Power cost ---
def benchmark_power(pay100, g, k, s):
    k_val = max(k / K_SCALER_POW, 1e-115)
    s_val = np.maximum(s / S_SCALER_POW + pay100, 1e-10)
    return (1 / g) * (np.log(s_val) - np.log(k_val))


def estimate_benchmark(dt):
    """Estimate benchmark model (Table 5, Panel A, NLS)."""
    mask = dt["dummy1"] == 1

    # Exponential
    st_exp = [0.015, 1.3e+11, 5.2e+3]
    sol_exp = opt.curve_fit(
        benchmark_exp,
        dt.loc[mask, "payoff_per_100"].values,
        dt.loc[mask, "buttonpresses_nearest_100"].values,
        p0=st_exp, maxfev=50000,
    )
    params_exp = sol_exp[0]
    se_exp = np.sqrt(np.diagonal(sol_exp[1]))

    gamma_exp = params_exp[0]
    k_exp = params_exp[1] / K_SCALER_EXP
    s_exp = params_exp[2] / S_SCALER_EXP

    # Power
    st_pow = [33, 6.0e+31, 4.9e+3]
    try:
        sol_pow = opt.curve_fit(
            benchmark_power,
            dt.loc[mask, "payoff_per_100"].values,
            dt.loc[mask, "logbuttonpresses_nearest_100"].values,
            p0=st_pow, maxfev=100000,
            method="trf",
        )
    except Exception:
        # Fallback: try least_squares directly
        sol_pow = opt.curve_fit(
            benchmark_power,
            dt.loc[mask, "payoff_per_100"].values,
            dt.loc[mask, "logbuttonpresses_nearest_100"].values,
            p0=st_pow, maxfev=100000,
        )
    params_pow = sol_pow[0]
    se_pow = np.sqrt(np.diagonal(sol_pow[1]))

    gamma_pow = params_pow[0]
    k_pow = params_pow[1] / K_SCALER_POW
    s_pow = params_pow[2] / S_SCALER_POW

    print("\n=== TABLE 5 PANEL A: Benchmark NLS ===")
    print(f"{'':20s} {'Exponential':>15s}  {'Power':>15s}")
    print(f"{'γ (curvature)':20s} {gamma_exp:15.6f}  {gamma_pow:15.6f}")
    print(f"{'k (cost level)':20s} {k_exp:15.6e}  {k_pow:15.6e}")
    print(f"{'s (intrinsic motiv)':20s} {s_exp:15.6e}  {s_pow:15.6e}")

    return {
        "exp": {"gamma": gamma_exp, "k": k_exp, "s": s_exp,
                "params_raw": params_exp, "se_raw": se_exp},
        "pow": {"gamma": gamma_pow, "k": k_pow, "s": s_pow,
                "params_raw": params_pow, "se_raw": se_pow},
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 5 PANEL B — All Parameters (except prob weighting)
# ═══════════════════════════════════════════════════════════════════════════════

def full_model_exp(xdata, g, k, s, alpha, a, gift, beta, delta):
    pay100, gd, dd, dw, paychar, dc = xdata
    k_val = k / K_SCALER_EXP
    s_val = (s / S_SCALER_EXP
             + gift * 0.4 * gd
             + (beta ** dd) * (delta ** dw) * pay100
             + alpha * paychar
             + a * 0.01 * dc)
    return (1 / g) * (np.log(np.maximum(s_val, 1e-20)) - np.log(k_val))


def full_model_power(xdata, g, k, s, alpha, a, gift, beta, delta):
    pay100, gd, dd, dw, paychar, dc = xdata
    k_val = max(k / K_SCALER_POW, 1e-115)
    s_val = np.maximum(
        s / S_SCALER_POW
        + gift * 0.4 * gd
        + (beta ** dd) * (delta ** dw) * pay100
        + alpha * paychar
        + a * 0.01 * dc,
        1e-10
    )
    return (1 / g) * (np.log(s_val) - np.log(k_val))


def estimate_full(dt):
    """Estimate full model (Table 5, Panel B, NLS)."""
    # Sample: piece rate + charity + gift + time discount (exclude prob weighting, gain/loss)
    valid_treatments = [
        "1.1", "1.2", "1.3", "1.4", "2",
        "3.1", "3.2", "4.1", "4.2", "10",
    ]
    mask = dt["treatment"].isin(valid_treatments)
    subset = dt[mask]

    xdata = np.array([
        subset["payoff_per_100"].values,
        subset["gift_dummy"].values,
        subset["delay_dummy"].values,
        subset["delay_wks"].values,
        subset["payoff_charity_per_100"].values,
        subset["dummy_charity"].values,
    ])

    # Exponential
    st_exp = [0.015, 1.3e+11, 5.2e+3, 0.5, 0.01, 0.01, 1.0, 0.99]
    try:
        sol_exp = opt.curve_fit(
            full_model_exp, xdata,
            subset["buttonpresses_nearest_100"].values,
            p0=st_exp, maxfev=100000,
        )
        p_exp = sol_exp[0]
        se_exp = np.sqrt(np.diagonal(sol_exp[1]))
        full_exp = {
            "gamma": p_exp[0], "k": p_exp[1] / K_SCALER_EXP,
            "s": p_exp[2] / S_SCALER_EXP, "alpha": p_exp[3],
            "a": p_exp[4], "gift": p_exp[5], "beta": p_exp[6], "delta": p_exp[7],
            "se_raw": se_exp,
        }
    except Exception as e:
        print(f"  Full model (exp) failed: {e}")
        full_exp = None

    # Power
    st_pow = [33, 6.0e+31, 4.9e+3, 0.5, 0.01, 0.01, 1.0, 0.99]
    try:
        sol_pow = opt.curve_fit(
            full_model_power, xdata,
            subset["logbuttonpresses_nearest_100"].values,
            p0=st_pow, maxfev=100000,
        )
        p_pow = sol_pow[0]
        se_pow = np.sqrt(np.diagonal(sol_pow[1]))
        full_pow = {
            "gamma": p_pow[0], "k": p_pow[1] / K_SCALER_POW,
            "s": p_pow[2] / S_SCALER_POW, "alpha": p_pow[3],
            "a": p_pow[4], "gift": p_pow[5], "beta": p_pow[6], "delta": p_pow[7],
            "se_raw": se_pow,
        }
    except Exception as e:
        print(f"  Full model (power) failed: {e}")
        full_pow = None

    print("\n=== TABLE 5 PANEL B: Full Model NLS ===")
    labels = ["γ", "k", "s", "α (altruism)", "a (warm glow)",
              "Δs_GE (gift)", "β (present bias)", "δ (discount)"]
    for lab, key in zip(labels, ["gamma", "k", "s", "alpha", "a", "gift", "beta", "delta"]):
        exp_val = f"{full_exp[key]:.6f}" if full_exp else "FAILED"
        pow_val = f"{full_pow[key]:.6f}" if full_pow else "FAILED"
        print(f"  {lab:22s}  Exp: {exp_val:>15s}  Pow: {pow_val:>15s}")

    return {"exp": full_exp, "pow": full_pow}


# ═══════════════════════════════════════════════════════════════════════════════
#  TABLE 6 — Probability Weighting
# ═══════════════════════════════════════════════════════════════════════════════

def prob_weight_exp(xdata, g, k, s, pi_w):
    """Probability weighting with curvature=1 (linear utility over piece rate)."""
    pay100, wd, prob = xdata
    k_val = k / K_SCALER_EXP
    s_val = s / S_SCALER_EXP + (pi_w ** wd) * prob * pay100
    return (1 / g) * (np.log(np.maximum(s_val, 1e-20)) - np.log(k_val))


def prob_weight_power(xdata, g, k, s, pi_w):
    pay100, wd, prob = xdata
    k_val = max(k / K_SCALER_POW, 1e-115)
    s_val = np.maximum(s / S_SCALER_POW + (pi_w ** wd) * prob * pay100, 1e-10)
    return (1 / g) * (np.log(s_val) - np.log(k_val))


def estimate_prob_weighting(dt):
    """Estimate probability weighting (Table 6, Panel A)."""
    mask = dt["treatment"].isin(["1.1", "1.2", "1.3", "6.1", "6.2"])
    subset = dt[mask]

    xdata = np.array([
        subset["payoff_per_100"].values,
        subset["weight_dummy"].values,
        subset["prob"].values,
    ])

    results = {}

    # Exponential
    st_exp = [0.015, 1.3e+11, 5.2e+3, 0.5]
    try:
        sol = opt.curve_fit(
            prob_weight_exp, xdata,
            subset["buttonpresses_nearest_100"].values,
            p0=st_exp, maxfev=50000,
        )
        p = sol[0]
        results["exp"] = {"gamma": p[0], "k": p[1] / K_SCALER_EXP,
                          "s": p[2] / S_SCALER_EXP, "pi_weight": p[3]}
    except Exception as e:
        print(f"  Prob weighting (exp) failed: {e}")
        results["exp"] = None

    # Power
    st_pow = [33, 6.0e+31, 4.9e+3, 0.5]
    try:
        sol = opt.curve_fit(
            prob_weight_power, xdata,
            subset["logbuttonpresses_nearest_100"].values,
            p0=st_pow, maxfev=50000,
        )
        p = sol[0]
        results["pow"] = {"gamma": p[0], "k": p[1] / K_SCALER_POW,
                          "s": p[2] / S_SCALER_POW, "pi_weight": p[3]}
    except Exception as e:
        print(f"  Prob weighting (power) failed: {e}")
        results["pow"] = None

    print("\n=== TABLE 6: Probability Weighting NLS ===")
    for spec in ["exp", "pow"]:
        r = results[spec]
        if r:
            print(f"  {spec.upper()}: γ={r['gamma']:.6f}, π_weight={r['pi_weight']:.6f}")

    return results


# ═══════════════════════════════════════════════════════════════════════════════
#  Save all results
# ═══════════════════════════════════════════════════════════════════════════════

def save_results(benchmark, full, prob_wt):
    """Save structural estimation results to CSV."""
    rows = []

    # Benchmark
    for spec in ["exp", "pow"]:
        b = benchmark[spec]
        rows.append({"Model": f"Benchmark ({spec})", "Method": "NLS",
                      "gamma": b["gamma"], "k": b["k"], "s": b["s"]})

    # Full model
    for spec in ["exp", "pow"]:
        f = full[spec]
        if f:
            rows.append({"Model": f"Full ({spec})", "Method": "NLS",
                          "gamma": f["gamma"], "k": f["k"], "s": f["s"],
                          "alpha": f["alpha"], "a": f["a"],
                          "gift": f["gift"], "beta": f["beta"], "delta": f["delta"]})

    # Prob weighting
    for spec in ["exp", "pow"]:
        pw = prob_wt[spec]
        if pw:
            rows.append({"Model": f"ProbWeight ({spec})", "Method": "NLS",
                          "gamma": pw["gamma"], "k": pw["k"], "s": pw["s"],
                          "pi_weight": pw["pi_weight"]})

    df_out = pd.DataFrame(rows)
    df_out.to_csv(os.path.join(TABLES_DIR, "table5_6_nls_results.csv"), index=False)
    print(f"\nResults saved to {os.path.join(TABLES_DIR, 'table5_6_nls_results.csv')}")
    return df_out


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    os.makedirs(TABLES_DIR, exist_ok=True)

    dt = load_and_prepare()
    print(f"Loaded {len(dt)} observations")

    benchmark = estimate_benchmark(dt)
    full = estimate_full(dt)
    prob_wt = estimate_prob_weighting(dt)
    save_results(benchmark, full, prob_wt)

    print("\n✓ NLS structural estimation complete.")
