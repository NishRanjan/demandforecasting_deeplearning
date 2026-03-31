"""
evaluation/metrics.py
─────────────────────
Pure metric functions. No state, no model imports, no config imports.
Each function operates on numpy arrays and returns a scalar float.
"""

from __future__ import annotations

import numpy as np


def wmape(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    weights: np.ndarray | None = None,
) -> float:
    """
    Weighted Mean Absolute Percentage Error.

    WMAPE = sum(|actual - forecast|) / sum(|actual|)

    Robust to zero actuals (returns NaN if all actuals are zero).
    When weights are provided, both numerator and denominator are weighted.
    """
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)

    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        numerator = np.sum(weights * np.abs(actuals - forecasts))
        denominator = np.sum(weights * np.abs(actuals))
    else:
        numerator = np.sum(np.abs(actuals - forecasts))
        denominator = np.sum(np.abs(actuals))

    if denominator < 1e-9:
        return float("nan")
    return float(numerator / denominator)


def r_squared(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """
    Coefficient of determination (R²).
    Returns NaN if actuals have zero variance.
    """
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)

    ss_res = np.sum((actuals - forecasts) ** 2)
    ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)

    if ss_tot < 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def bias(actuals: np.ndarray, forecasts: np.ndarray) -> float:
    """
    Signed bias as a fraction of total actuals.
    Positive = over-forecast, negative = under-forecast.

    bias = sum(forecast - actual) / sum(|actual|)
    """
    actuals = np.asarray(actuals, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)

    denominator = np.sum(np.abs(actuals))
    if denominator < 1e-9:
        return float("nan")
    return float(np.sum(forecasts - actuals) / denominator)


def quantile_coverage(
    actuals: np.ndarray,
    q_low: np.ndarray,
    q_high: np.ndarray,
) -> float:
    """
    Fraction of actuals that fall within the [q_low, q_high] prediction interval.
    Values range from 0.0 to 1.0.
    """
    actuals = np.asarray(actuals, dtype=float)
    q_low = np.asarray(q_low, dtype=float)
    q_high = np.asarray(q_high, dtype=float)

    inside = np.sum((actuals >= q_low) & (actuals <= q_high))
    return float(inside / len(actuals))


def compute_all_metrics(
    actuals: np.ndarray,
    forecasts: np.ndarray,
    q10: np.ndarray | None = None,
    q90: np.ndarray | None = None,
) -> dict:
    """
    Compute all metrics at once. Returns a dict ready for DataFrame insertion.
    Pass q10/q90=None for models that don't produce prediction intervals.
    """
    result = {
        "wmape": wmape(actuals, forecasts),
        "r2": r_squared(actuals, forecasts),
        "bias": bias(actuals, forecasts),
    }
    if q10 is not None and q90 is not None:
        result["quantile_coverage"] = quantile_coverage(actuals, q10, q90)
    else:
        result["quantile_coverage"] = float("nan")
    return result
