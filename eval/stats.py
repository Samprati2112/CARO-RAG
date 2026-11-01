"""
Statistical rigor module:
- Bootstrap 95% Confidence Intervals (>=1000 resamples) for every headline metric
- Paired bootstrap hypothesis testing for CARO vs each baseline on key metrics
- p-values, effect sizes (Cohen's d), and strict verdict determination (CI excludes zero).
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def compute_bootstrap_ci(
    values: List[float],
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """
    Computes mean and percentile bootstrap confidence interval for a list of values.
    Returns: {'mean': float, 'ci_lower': float, 'ci_upper': float, 'std': float}
    """
    if not values:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 1:
        v = float(arr[0])
        return {"mean": v, "ci_lower": v, "ci_upper": v, "std": 0.0}

    rng = np.random.RandomState(seed)
    boot_means = np.empty(n_resamples, dtype=np.float64)

    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_means[i] = np.mean(arr[indices])

    alpha = 1.0 - ci
    lower_pct = (alpha / 2.0) * 100.0
    upper_pct = (1.0 - alpha / 2.0) * 100.0

    ci_lower = float(np.percentile(boot_means, lower_pct))
    ci_upper = float(np.percentile(boot_means, upper_pct))
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if n > 1 else 0.0

    return {
        "mean": round(mean_val, 4),
        "ci_lower": round(ci_lower, 4),
        "ci_upper": round(ci_upper, 4),
        "std": round(std_val, 4)
    }


def paired_bootstrap_test(
    caro_values: List[float],
    baseline_values: List[float],
    n_resamples: int = 1000,
    seed: int = 42,
    higher_is_better: bool = True
) -> Dict[str, Any]:
    """
    Runs paired bootstrap test between CARO and a baseline system.
    Returns:
        - diff_mean: mean(CARO - Baseline)
        - diff_ci_lower: 95% CI lower bound of difference
        - diff_ci_upper: 95% CI upper bound of difference
        - p_value: two-sided bootstrap p-value against null hypothesis (diff == 0)
        - effect_size: Cohen's d for paired differences
        - verdict: 'CARO significantly better' | 'CARO significantly worse' | 'No significant difference'
    """
    if len(caro_values) != len(baseline_values):
        raise ValueError(f"Length mismatch: CARO ({len(caro_values)}) vs Baseline ({len(baseline_values)})")

    if not caro_values:
        return {
            "diff_mean": 0.0, "diff_ci_lower": 0.0, "diff_ci_upper": 0.0,
            "p_value": 1.0, "effect_size": 0.0, "verdict": "No data"
        }

    c_arr = np.asarray(caro_values, dtype=np.float64)
    b_arr = np.asarray(baseline_values, dtype=np.float64)
    diff = c_arr - b_arr
    n = len(diff)

    obs_diff = float(np.mean(diff))
    diff_std = float(np.std(diff, ddof=1)) if n > 1 else 1e-9
    effect_size = float(obs_diff / diff_std) if diff_std > 1e-9 else 0.0

    rng = np.random.RandomState(seed)
    boot_diffs = np.empty(n_resamples, dtype=np.float64)
    for i in range(n_resamples):
        indices = rng.randint(0, n, size=n)
        boot_diffs[i] = np.mean(diff[indices])

    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))

    # Two-sided p-value under null hypothesis (centering bootstrap distribution at 0)
    null_boot_diffs = boot_diffs - obs_diff
    p_value = float(np.mean(np.abs(null_boot_diffs) >= np.abs(obs_diff)))

    # Strict rule: Do not claim better/worse unless the 95% CI excludes zero
    excludes_zero = (ci_lower > 0 and ci_upper > 0) or (ci_lower < 0 and ci_upper < 0)

    if higher_is_better:
        if excludes_zero and obs_diff > 0:
            verdict = "CARO significantly better"
        elif excludes_zero and obs_diff < 0:
            verdict = "CARO significantly worse"
        else:
            verdict = "No significant difference"
    else:  # Lower is better (e.g. hallucination rate, latency)
        if excludes_zero and obs_diff < 0:
            verdict = "CARO significantly better"
        elif excludes_zero and obs_diff > 0:
            verdict = "CARO significantly worse"
        else:
            verdict = "No significant difference"

    return {
        "diff_mean": round(obs_diff, 4),
        "diff_ci_lower": round(ci_lower, 4),
        "diff_ci_upper": round(ci_upper, 4),
        "p_value": round(p_value, 4),
        "effect_size": round(effect_size, 4),
        "verdict": verdict
    }
