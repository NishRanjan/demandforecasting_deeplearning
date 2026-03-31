"""
evaluation/comparison.py
─────────────────────────
Aggregates backtest results from all models and computes a unified
comparison report.

Main entry point: compare_models()
Inputs : dict mapping model_name → backtest DataFrame (standard schema)
Outputs: (summary_df, per_series_df)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.metrics import compute_all_metrics


def compare_models(
    results_dict: dict[str, pd.DataFrame],
    grain_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute evaluation metrics for all models and return comparison tables.

    Parameters
    ----------
    results_dict : dict of { model_name: backtest_df }
        Each backtest_df must have columns:
            [date_col, *grain_cols, "forecast", "q10", "q90",
             "model_name", "actual", "series_segment", "backtest_cutoff"]
    grain_cols : list of grain column names (e.g. ["sku", "state"])
                 If None, inferred from the DataFrames.

    Returns
    -------
    summary_df    : One row per model — aggregate metrics + best_series_count
    per_series_df : One row per (model, grain, backtest_cutoff) — detailed metrics
    """
    if not results_dict:
        return pd.DataFrame(), pd.DataFrame()

    # Infer grain_cols from data if not provided
    if grain_cols is None:
        sample = next(iter(results_dict.values()))
        exclude = {"forecast", "q10", "q90", "model_name", "actual",
                   "series_segment", "backtest_cutoff"}
        # Try to detect grain cols as non-date, non-metric columns
        grain_cols = [c for c in sample.columns
                      if c not in exclude and not pd.api.types.is_datetime64_any_dtype(sample[c])]

    per_series_records = []
    for model_name, bt_df in results_dict.items():
        records = _per_series_metrics(bt_df, grain_cols, model_name)
        per_series_records.extend(records)

    per_series_df = pd.DataFrame(per_series_records)

    summary_df = _build_summary(per_series_df, list(results_dict.keys()))

    return summary_df, per_series_df


def _per_series_metrics(
    bt_df: pd.DataFrame,
    grain_cols: list[str],
    model_name: str,
) -> list[dict]:
    """Compute metrics for each (grain, backtest_cutoff) combination."""
    records = []

    group_cols = grain_cols + ["backtest_cutoff"] if "backtest_cutoff" in bt_df.columns else grain_cols
    group_cols = [c for c in group_cols if c in bt_df.columns]

    for keys, grp in bt_df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)

        actuals = grp["actual"].values
        forecasts = grp["forecast"].values

        # Skip rows where actual is missing (series not in that window)
        mask = ~np.isnan(actuals) & ~np.isnan(forecasts)
        if mask.sum() == 0:
            continue

        q10 = grp["q10"].values if "q10" in grp.columns else None
        q90 = grp["q90"].values if "q90" in grp.columns else None

        # Only pass quantiles if they're not all NaN
        q10_valid = q10 if (q10 is not None and not np.all(np.isnan(q10))) else None
        q90_valid = q90 if (q90 is not None and not np.all(np.isnan(q90))) else None

        metrics = compute_all_metrics(
            actuals[mask], forecasts[mask],
            q10=q10_valid[mask] if q10_valid is not None else None,
            q90=q90_valid[mask] if q90_valid is not None else None,
        )

        record = {"model_name": model_name}
        for i, gc in enumerate(group_cols):
            record[gc] = keys[i]

        # Series segment (head/tail) — take first value in group
        if "series_segment" in grp.columns:
            record["series_segment"] = grp["series_segment"].iloc[0]

        record.update(metrics)
        records.append(record)

    return records


def _build_summary(per_series_df: pd.DataFrame, model_names: list[str]) -> pd.DataFrame:
    """
    Aggregate per-series metrics to one row per model.
    Also computes best_series_count = #series where this model had lowest WMAPE.
    """
    if per_series_df.empty:
        return pd.DataFrame()

    # Aggregate mean metrics per model
    metric_cols = ["wmape", "r2", "bias", "quantile_coverage"]
    metric_cols = [c for c in metric_cols if c in per_series_df.columns]

    agg = per_series_df.groupby("model_name")[metric_cols].agg(
        ["mean", "median"]
    )
    agg.columns = [f"{m}_{stat}" for m, stat in agg.columns]
    agg = agg.reset_index()

    # Count series per model
    n_series = per_series_df.groupby("model_name").size().rename("n_evaluations").reset_index()
    agg = agg.merge(n_series, on="model_name")

    # Best-series count: for each series, which model had lowest WMAPE?
    if "wmape" in per_series_df.columns:
        # Identify grain columns (exclude model_name, metrics, segment, cutoff)
        exclude = {"model_name", "wmape", "r2", "bias", "quantile_coverage",
                   "series_segment", "backtest_cutoff"}
        id_cols = [c for c in per_series_df.columns if c not in exclude]

        if id_cols:
            best = (
                per_series_df.dropna(subset=["wmape"])
                .loc[per_series_df.groupby(id_cols)["wmape"].idxmin()]
                ["model_name"]
                .value_counts()
                .rename("best_series_count")
                .reset_index()
            )
            best.columns = ["model_name", "best_series_count"]
            agg = agg.merge(best, on="model_name", how="left")
            agg["best_series_count"] = agg["best_series_count"].fillna(0).astype(int)

    # Sort by primary metric (wmape_mean ascending)
    if "wmape_mean" in agg.columns:
        agg = agg.sort_values("wmape_mean")

    return agg.reset_index(drop=True)


def print_comparison(summary_df: pd.DataFrame) -> None:
    """Pretty-print the summary table to console."""
    if summary_df.empty:
        print("No comparison results available.")
        return

    print("\n" + "=" * 70)
    print("  MODEL COMPARISON SUMMARY")
    print("=" * 70)

    display_cols = [c for c in [
        "model_name", "wmape_mean", "wmape_median",
        "r2_mean", "bias_mean", "quantile_coverage_mean",
        "n_evaluations", "best_series_count"
    ] if c in summary_df.columns]

    fmt = summary_df[display_cols].copy()
    for col in ["wmape_mean", "wmape_median", "bias_mean"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")
    for col in ["r2_mean", "quantile_coverage_mean"]:
        if col in fmt.columns:
            fmt[col] = fmt[col].map(lambda x: f"{x:.4f}" if pd.notna(x) else "N/A")

    print(fmt.to_string(index=False))
    print("=" * 70 + "\n")
