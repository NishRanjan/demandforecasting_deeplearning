"""
utils/date_utils.py
───────────────────
Date arithmetic for backtest windows and M+1 latency convention.

M+1 convention (as used in the source notebooks):
  - Data is available up to month M but with 1-month latency.
  - So at decision time, we train on data through M, but the FIRST
    forecastable month is M+2 (one latency month skipped).
  - forecast_horizon=2 means we forecast months M+2 and M+3.
"""

from __future__ import annotations

from typing import List

import pandas as pd


def get_backtest_cutoffs(
    df: pd.DataFrame,
    date_col: str,
    n_windows: int,
    latency_months: int = 1,
) -> List[pd.Timestamp]:
    """
    Returns a list of train-end cutoff dates for rolling-origin backtest.

    The last available month in df is held out as the most-recent forward
    forecast period. Backtest cutoffs step back from there.

    Parameters
    ----------
    df : DataFrame with a date_col column
    date_col : name of the date column
    n_windows : number of rolling windows (e.g. 3 → last 3 months)
    latency_months : data latency (1 for M+1 convention)

    Returns
    -------
    List of cutoff pd.Timestamps, ordered oldest → newest.
    E.g. for n_windows=3 and latest month = Dec 2025:
        [Sep 2025, Oct 2025, Nov 2025]
    """
    latest = df[date_col].max()
    # Step back n_windows from the month just before the latest
    # The latest full data month used for evaluation ends at latest - latency
    base = latest - pd.DateOffset(months=latency_months)
    cutoffs = [
        base - pd.DateOffset(months=i)
        for i in range(n_windows - 1, -1, -1)
    ]
    return [pd.Timestamp(c) for c in cutoffs]


def apply_latency_offset(cutoff_date: pd.Timestamp, latency_months: int) -> pd.Timestamp:
    """
    Given a train cutoff date, return the first forecastable date.

    With latency_months=1 (M+1):
      cutoff = Oct 2025  →  first forecast = Dec 2025  (skip Nov, the latency month)
    """
    return cutoff_date + pd.DateOffset(months=latency_months + 1)


def get_forecast_dates(
    start_date: pd.Timestamp,
    horizon: int,
    freq: str = "MS",
) -> pd.DatetimeIndex:
    """
    Returns a DatetimeIndex of `horizon` future periods starting at start_date.
    """
    return pd.date_range(start=start_date, periods=horizon, freq=freq)


def split_train_context(
    df: pd.DataFrame,
    date_col: str,
    cutoff_date: pd.Timestamp,
    encoder_length: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split df into:
      - train_df  : all rows with date <= cutoff_date
      - context_df: last `encoder_length` months of train_df (for inference context)

    Returns (train_df, context_df).
    """
    train_df = df[df[date_col] <= cutoff_date].copy()
    context_start = cutoff_date - pd.DateOffset(months=encoder_length - 1)
    context_df = train_df[train_df[date_col] >= context_start].copy()
    return train_df, context_df
