"""
data/preprocessor.py
────────────────────
Cleaning, outlier handling, imputation, log1p transform, and series
segmentation (head/tail).

All transformations are config-driven. The output DataFrame is a
cleaned version of the input with two added columns:
  - series_segment : "head" or "tail"
  - target_raw     : copy of original target before log1p (useful for reporting)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def preprocess(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Full preprocessing pipeline. Returns cleaned DataFrame.
    """
    cfg = config["preprocessing"]
    data_cfg = config["data"]

    date_col = data_cfg["date_col"]
    target_col = data_cfg["target_col"]
    grain_cols = data_cfg.get("grain_cols", [])

    df = df.copy()

    # 1. Preserve original target
    df["target_raw"] = df[target_col]

    # 2. Outlier clipping (per grain group)
    method = cfg.get("outlier_method", "iqr")
    if method != "none":
        df = _clip_outliers(df, target_col, grain_cols, method, cfg.get("iqr_multiplier", 1.5))

    # 3. Imputation of numeric columns
    imputation = cfg.get("imputation_method", "forward_fill")
    df = _impute(df, date_col, grain_cols, imputation)

    # 4. Log1p transform on target
    if cfg.get("log1p_transform", True):
        df[target_col] = np.log1p(df[target_col].clip(lower=0))

    # 5. Head / tail segmentation
    min_history = cfg.get("min_history_months", 12)
    df = _tag_segment(df, date_col, grain_cols, min_history)

    print(f"[preprocessor] After preprocessing: {len(df):,} rows")
    seg_counts = df["series_segment"].value_counts().to_dict()
    print(f"[preprocessor] Series segments: {seg_counts}")
    return df


# ─────────────────────────────────────────────────────────────────────────────

def _clip_outliers(
    df: pd.DataFrame,
    target_col: str,
    grain_cols: list,
    method: str,
    iqr_multiplier: float,
) -> pd.DataFrame:
    """Clip target column to [lower, upper] bounds computed per grain group."""

    def _bounds_iqr(series: pd.Series, multiplier: float):
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        return q1 - multiplier * iqr, q3 + multiplier * iqr

    def _bounds_zscore(series: pd.Series, threshold: float = 3.0):
        mean = series.mean()
        std = series.std()
        return mean - threshold * std, mean + threshold * std

    if grain_cols:
        def clip_group(grp):
            if method == "iqr":
                lb, ub = _bounds_iqr(grp[target_col], iqr_multiplier)
            else:
                lb, ub = _bounds_zscore(grp[target_col])
            grp[target_col] = grp[target_col].clip(lower=lb, upper=ub)
            return grp

        df = df.groupby(grain_cols, group_keys=False).apply(clip_group)
    else:
        if method == "iqr":
            lb, ub = _bounds_iqr(df[target_col], iqr_multiplier)
        else:
            lb, ub = _bounds_zscore(df[target_col])
        df[target_col] = df[target_col].clip(lower=lb, upper=ub)

    return df


def _impute(
    df: pd.DataFrame,
    date_col: str,
    grain_cols: list,
    method: str,
) -> pd.DataFrame:
    """Impute missing values in all numeric columns (excluding grain cols and date)."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if grain_cols:
        def impute_group(grp):
            grp = grp.sort_values(date_col)
            for col in numeric_cols:
                if grp[col].isna().any():
                    if method == "forward_fill":
                        grp[col] = grp[col].fillna(method="ffill").fillna(method="bfill")
                    elif method == "interpolate":
                        grp[col] = grp[col].interpolate(method="linear", limit_direction="both")
                    elif method == "zero":
                        grp[col] = grp[col].fillna(0)
            return grp

        df = df.groupby(grain_cols, group_keys=False).apply(impute_group)
    else:
        for col in numeric_cols:
            if method == "forward_fill":
                df[col] = df[col].fillna(method="ffill").fillna(method="bfill")
            elif method == "interpolate":
                df[col] = df[col].interpolate(method="linear", limit_direction="both")
            elif method == "zero":
                df[col] = df[col].fillna(0)

    return df


def _tag_segment(
    df: pd.DataFrame,
    date_col: str,
    grain_cols: list,
    min_history_months: int,
) -> pd.DataFrame:
    """
    Tag each row with 'head' or 'tail' based on whether the series has
    at least min_history_months of data.

    - head: >= min_history_months — mature product, reliable history
    - tail: < min_history_months — new product, sparse data
    """
    if grain_cols:
        history_counts = df.groupby(grain_cols)[date_col].count().reset_index()
        history_counts.columns = grain_cols + ["_n_months"]
        df = df.merge(history_counts, on=grain_cols, how="left")
        df["series_segment"] = np.where(df["_n_months"] >= min_history_months, "head", "tail")
        df = df.drop(columns=["_n_months"])
    else:
        n_months = df[date_col].nunique()
        df["series_segment"] = "head" if n_months >= min_history_months else "tail"

    return df
