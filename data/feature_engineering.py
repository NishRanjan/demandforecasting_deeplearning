"""
data/feature_engineering.py
────────────────────────────
Generate derived features: lags, rolling statistics, and calendar features.

All column names come from config — nothing is hardcoded here.
Feature engineering runs per grain group via groupby/apply.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Add calendar features and generate lag columns specified in config.
    Returns the fully featured DataFrame ready to pass to models.
    """
    data_cfg = config["data"]
    feat_cfg = config.get("features", {})

    date_col = data_cfg["date_col"]
    target_col = data_cfg["target_col"]
    grain_cols = data_cfg.get("grain_cols", [])

    df = df.copy()

    # 1. Calendar features (always added — they're in time_varying_known)
    df = _add_calendar_features(df, date_col)

    # 2. Lag columns specified in config
    lag_col_names = feat_cfg.get("continuous", {}).get("lag_cols", [])
    if lag_col_names:
        df = _add_lags_from_config(df, date_col, target_col, grain_cols, lag_col_names)

    # 3. Rolling statistics on target (3-month and 6-month moving average)
    if grain_cols:
        df = _add_rolling_stats(df, date_col, target_col, grain_cols)

    print(f"[feature_engineering] Feature matrix: {df.shape[1]} columns")
    return df


# ─────────────────────────────────────────────────────────────────────────────

def _add_calendar_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add month (1-12) and quarterly bucket (Q1-Q4) columns."""
    df["month"] = df[date_col].dt.month
    df["qtr_bucket"] = df[date_col].dt.quarter.map(
        {1: "Q1", 2: "Q2", 3: "Q3", 4: "Q4"}
    )
    return df


def _add_lags_from_config(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    grain_cols: list,
    lag_col_names: list,
) -> pd.DataFrame:
    """
    Generate lag columns from the target.

    The config provides the *names* of the lag columns (e.g. "btl_lag1").
    We infer the lag number from the trailing digit(s) in the name.

    If the source data already has these columns populated, this step
    will skip generating them (only creates missing ones).
    """
    for col_name in lag_col_names:
        if col_name in df.columns:
            continue  # Already present in source data

        # Extract lag integer from trailing digits, e.g. "btl_lag3" → 3
        digits = "".join(filter(str.isdigit, col_name.split("lag")[-1]))
        if not digits:
            continue
        lag_n = int(digits)

        if grain_cols:
            df[col_name] = df.groupby(grain_cols)[target_col].shift(lag_n)
        else:
            df[col_name] = df[target_col].shift(lag_n)

    return df


def _add_rolling_stats(
    df: pd.DataFrame,
    date_col: str,
    target_col: str,
    grain_cols: list,
) -> pd.DataFrame:
    """Add 3-month and 6-month rolling means per grain (shifted to avoid leakage)."""
    def _rolling(grp):
        grp = grp.sort_values(date_col)
        shifted = grp[target_col].shift(1)
        grp["target_rolling_3m"] = shifted.rolling(3, min_periods=1).mean()
        grp["target_rolling_6m"] = shifted.rolling(6, min_periods=1).mean()
        return grp

    df = df.groupby(grain_cols, group_keys=False).apply(_rolling)
    return df
