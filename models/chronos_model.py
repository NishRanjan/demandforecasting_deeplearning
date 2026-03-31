"""
models/chronos_model.py
────────────────────────
Amazon Chronos-2 forecaster wrapper.

Chronos is a pre-trained foundation model — fit() just loads the pipeline.
Inference converts each grain's history to a torch.Tensor context window
and runs batch prediction to get quantile forecasts.

M+1 convention is handled in backtest() via date_utils.
"""

from __future__ import annotations

import warnings
from typing import List

import numpy as np
import pandas as pd
import torch

from models.base import BaseForecaster
from utils.date_utils import (
    apply_latency_offset,
    get_backtest_cutoffs,
    get_forecast_dates,
)


class ChronosForecaster(BaseForecaster):

    @property
    def name(self) -> str:
        return "chronos"

    def fit(self, train_df: pd.DataFrame) -> None:
        """Load the Chronos pipeline. No training — weights are pre-trained."""
        try:
            from chronos import BaseChronosPipeline
        except ImportError:
            raise ImportError(
                "chronos package not found. Install with: pip install chronos-forecasting"
            )

        model_name = self.model_cfg.get("model_name", "amazon/chronos-t5-large")
        device = self.model_cfg.get("device", "cpu")

        print(f"[chronos] Loading pipeline: {model_name} on {device}")
        self._pipeline = BaseChronosPipeline.from_pretrained(
            model_name,
            device_map=device,
            torch_dtype=torch.bfloat16,
        )
        self._is_fitted = True
        print("[chronos] Pipeline ready.")

    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Forecast `horizon` steps for all grain combinations in context_df.

        Parameters
        ----------
        context_df : history up to the cutoff date (all dates, all grains)
        horizon    : number of future months to forecast

        Returns
        -------
        DataFrame with standard output schema.
        """
        self._check_fitted()

        quantiles = self.model_cfg.get("quantiles", [0.1, 0.5, 0.9])
        batch_size = self.model_cfg.get("batch_size", 128)
        context_length = self.model_cfg.get("context_length", 24)

        grain_cols = self.grain_cols
        date_col = self.date_col
        target_col = self.target_col
        freq = self.data_cfg.get("freq", "MS")

        # Group by grain, collect context tensors
        groups = list(context_df.groupby(grain_cols))
        all_records = []

        for batch_start in range(0, len(groups), batch_size):
            batch = groups[batch_start: batch_start + batch_size]

            context_tensors: List[torch.Tensor] = []
            grain_keys = []
            forecast_starts = []

            for grain_key, grp in batch:
                grp = grp.sort_values(date_col)
                series = grp[target_col].values[-context_length:]
                context_tensors.append(torch.tensor(series, dtype=torch.float32))

                # Forecast start = next period after last date in context
                last_date = grp[date_col].max()
                forecast_start = last_date + pd.DateOffset(months=1)
                forecast_starts.append(pd.Timestamp(forecast_start))

                grain_keys.append(
                    grain_key if isinstance(grain_key, tuple) else (grain_key,)
                )

            with torch.no_grad():
                forecast_tensor, forecast_quantile_levels, _ = self._pipeline.predict_quantiles(
                    context=context_tensors,
                    prediction_length=horizon,
                    quantile_levels=quantiles,
                )
            # forecast_tensor shape: (batch, n_quantiles, horizon)

            q10_idx = quantiles.index(0.1) if 0.1 in quantiles else 0
            q50_idx = quantiles.index(0.5) if 0.5 in quantiles else len(quantiles) // 2
            q90_idx = quantiles.index(0.9) if 0.9 in quantiles else -1

            for i, (grain_key, fstart) in enumerate(zip(grain_keys, forecast_starts)):
                forecast_dates = get_forecast_dates(fstart, horizon, freq)
                for t, fdate in enumerate(forecast_dates):
                    record = {date_col: fdate}
                    for j, gc in enumerate(grain_cols):
                        record[gc] = grain_key[j]
                    record["forecast"] = float(forecast_tensor[i, q50_idx, t])
                    record["q10"] = float(forecast_tensor[i, q10_idx, t])
                    record["q90"] = float(forecast_tensor[i, q90_idx, t])
                    record["model_name"] = self.name
                    all_records.append(record)

        return pd.DataFrame(all_records)

    def backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling-origin backtest with M+1 latency convention."""
        n_windows = self.backtest_cfg["n_rolling_windows"]
        latency = self.backtest_cfg["data_latency_months"]
        horizon = self.backtest_cfg["forecast_horizon"]

        cutoffs = get_backtest_cutoffs(df, self.date_col, n_windows, latency)
        print(f"[chronos] Backtest cutoffs: {[str(c.date()) for c in cutoffs]}")

        all_results = []

        for cutoff in cutoffs:
            train_df = df[df[self.date_col] <= cutoff].copy()
            self.fit(train_df)

            forecast_start = apply_latency_offset(cutoff, latency)
            forecast_df = self.predict(train_df, horizon)

            # Attach actuals
            actuals_df = df[
                (df[self.date_col] >= forecast_start) &
                (df[self.date_col] < forecast_start + pd.DateOffset(months=horizon))
            ][[self.date_col] + self.grain_cols + [self.target_col, "series_segment"]].copy()
            actuals_df = actuals_df.rename(columns={self.target_col: "actual"})

            merged = forecast_df.merge(actuals_df, on=[self.date_col] + self.grain_cols, how="left")
            merged["backtest_cutoff"] = cutoff
            all_results.append(merged)

        return pd.concat(all_results, ignore_index=True)
