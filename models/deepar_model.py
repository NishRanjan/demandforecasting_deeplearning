"""
models/deepar_model.py
──────────────────────
DeepAR forecaster using pytorch-forecasting.

DeepAR is a probabilistic RNN-based model that naturally produces
prediction intervals (via distributional outputs). It follows the
same TimeSeriesDataSet + Lightning Trainer pattern as TFT and NHiTS.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from models.base import BaseForecaster
from utils.date_utils import (
    apply_latency_offset,
    get_backtest_cutoffs,
    get_forecast_dates,
)


class DeepARForecaster(BaseForecaster):

    @property
    def name(self) -> str:
        return "deepar"

    def fit(self, train_df: pd.DataFrame) -> None:
        """Build TimeSeriesDataSet and train DeepAR with Lightning."""
        try:
            from pytorch_forecasting import TimeSeriesDataSet, DeepAR
            from pytorch_forecasting.data import GroupNormalizer
            from pytorch_forecasting.metrics import NormalDistributionLoss
            import lightning.pytorch as pl
            from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
        except ImportError:
            raise ImportError(
                "pytorch-forecasting and lightning are required. "
                "Install with: pip install pytorch-forecasting lightning"
            )

        cfg = self.model_cfg
        max_encoder_length = cfg.get("max_encoder_length", 24)
        max_prediction_length = cfg.get("max_prediction_length", 2)

        group_ids = self.grain_cols if self.grain_cols else ["_group"]
        df = train_df.copy()
        if not self.grain_cols:
            df["_group"] = "all"

        df = df.sort_values([self.date_col] + group_ids)
        min_date = df[self.date_col].min()
        df["time_idx"] = ((df[self.date_col].dt.year - min_date.year) * 12 +
                          (df[self.date_col].dt.month - min_date.month))

        training_cutoff = df["time_idx"].max() - max_prediction_length

        self._training_ds = TimeSeriesDataSet(
            df[df["time_idx"] <= training_cutoff],
            time_idx="time_idx",
            target=self.target_col,
            group_ids=group_ids,
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_reals=[self.target_col],
            time_varying_known_reals=["time_idx"],
            target_normalizer=GroupNormalizer(groups=group_ids, transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
        )

        val_ds = TimeSeriesDataSet.from_dataset(
            self._training_ds, df, predict=True, stop_randomization=True
        )

        batch_size = cfg.get("batch_size", 64)
        num_workers = cfg.get("num_workers", 0)
        train_loader = self._training_ds.to_dataloader(train=True, batch_size=batch_size, num_workers=num_workers)
        val_loader = val_ds.to_dataloader(train=False, batch_size=batch_size * 2, num_workers=num_workers)

        self._model = DeepAR.from_dataset(
            self._training_ds,
            learning_rate=cfg.get("learning_rate", 0.001),
            hidden_size=cfg.get("hidden_size", 32),
            rnn_layers=cfg.get("rnn_layers", 2),
            dropout=cfg.get("dropout", 0.1),
            loss=NormalDistributionLoss(),
            log_interval=10,
        )

        trainer = pl.Trainer(
            max_epochs=cfg.get("max_epochs", 50),
            accelerator=cfg.get("accelerator", "cpu"),
            devices=cfg.get("devices", 1),
            gradient_clip_val=cfg.get("gradient_clip_val", 0.1),
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=cfg.get("early_stopping_patience", 10), mode="min"),
                LearningRateMonitor(),
            ],
            enable_progress_bar=True,
        )

        print(f"[deepar] Training DeepAR: encoder={max_encoder_length}, horizon={max_prediction_length}")
        trainer.fit(self._model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self._min_date = min_date
        self._group_ids = group_ids
        self._is_fitted = True
        print("[deepar] Training complete.")

    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Generate forecasts with prediction intervals.
        DeepAR's NormalDistributionLoss allows sampling for quantiles.
        """
        self._check_fitted()

        df = context_df.copy()
        if not self.grain_cols:
            df["_group"] = "all"

        df["time_idx"] = ((df[self.date_col].dt.year - self._min_date.year) * 12 +
                          (df[self.date_col].dt.month - self._min_date.month))

        from pytorch_forecasting import TimeSeriesDataSet
        predict_ds = TimeSeriesDataSet.from_dataset(
            self._training_ds, df, predict=True, stop_randomization=True
        )
        loader = predict_ds.to_dataloader(train=False, batch_size=self.model_cfg.get("batch_size", 64))

        # Use mode (mean) for point forecast; sample for quantiles
        raw_preds, idx = self._model.predict(loader, return_index=True, return_x=False)

        results = []
        max_date = context_df[self.date_col].max()
        freq = self.data_cfg.get("freq", "MS")

        for i, row in idx.iterrows():
            fstart = max_date + pd.DateOffset(months=1)
            forecast_dates = get_forecast_dates(pd.Timestamp(fstart), horizon, freq)
            for t, fdate in enumerate(forecast_dates):
                record = {self.date_col: fdate}
                for gc in self.grain_cols:
                    record[gc] = row.get(gc, "all")
                record["forecast"] = float(raw_preds[i, t]) if t < raw_preds.shape[1] else np.nan
                # DeepAR quantile intervals require sampling — set to NaN for now
                # Use model.predict_quantiles() for full interval support
                record["q10"] = np.nan
                record["q90"] = np.nan
                record["model_name"] = self.name
                results.append(record)

        return pd.DataFrame(results)

    def backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """Rolling-origin backtest with M+1 latency convention."""
        n_windows = self.backtest_cfg["n_rolling_windows"]
        latency = self.backtest_cfg["data_latency_months"]
        horizon = self.backtest_cfg["forecast_horizon"]

        cutoffs = get_backtest_cutoffs(df, self.date_col, n_windows, latency)
        print(f"[deepar] Backtest cutoffs: {[str(c.date()) for c in cutoffs]}")

        all_results = []
        for cutoff in cutoffs:
            train_df = df[df[self.date_col] <= cutoff].copy()
            self.fit(train_df)

            forecast_start = apply_latency_offset(cutoff, latency)
            forecast_df = self.predict(train_df, horizon)

            actuals_df = df[
                (df[self.date_col] >= forecast_start) &
                (df[self.date_col] < forecast_start + pd.DateOffset(months=horizon))
            ][[self.date_col] + self.grain_cols + [self.target_col, "series_segment"]].copy()
            actuals_df = actuals_df.rename(columns={self.target_col: "actual"})

            merged = forecast_df.merge(actuals_df, on=[self.date_col] + self.grain_cols, how="left")
            merged["backtest_cutoff"] = cutoff
            all_results.append(merged)

        return pd.concat(all_results, ignore_index=True)
