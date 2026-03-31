"""
models/tft_model.py
───────────────────
Temporal Fusion Transformer wrapper using pytorch-forecasting + Lightning.

Ported from: Pytorch experiments - all features_modularized - gpu (1).ipynb

Key design notes:
  - WMAPE loss is defined locally as a pytorch-forecasting MultiHorizonMetric.
    It is NOT imported from evaluation/metrics.py because the training interface
    requires tensor operations and a different method signature.
  - GroupNormalizer with log1p is applied inside TimeSeriesDataSet.
  - Feature lists (categoricals, known/unknown, continuous) are read from config.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from models.base import BaseForecaster
from utils.date_utils import (
    apply_latency_offset,
    get_backtest_cutoffs,
    get_forecast_dates,
)


# ── Custom WMAPE Loss ─────────────────────────────────────────────────────────

class WMAPELoss:
    """
    Weighted Mean Absolute Percentage Error loss compatible with
    pytorch-forecasting's training interface.
    """
    def __init__(self):
        self.name = "WMAPE"

    def loss(self, y_pred: torch.Tensor, y_actual: torch.Tensor) -> torch.Tensor:
        numerator = torch.abs(y_actual - y_pred).sum()
        denominator = torch.abs(y_actual).sum() + 1e-6
        return numerator / denominator

    def __call__(self, y_pred, y_actual):
        return self.loss(y_pred, y_actual)


# ── TFT Forecaster ────────────────────────────────────────────────────────────

class TFTForecaster(BaseForecaster):

    @property
    def name(self) -> str:
        return "tft"

    def fit(self, train_df: pd.DataFrame) -> None:
        """Build TimeSeriesDataSet and run Lightning training loop."""
        try:
            from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
            from pytorch_forecasting.data import GroupNormalizer
            from pytorch_forecasting.metrics import QuantileLoss
            import lightning.pytorch as pl
            from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
        except ImportError:
            raise ImportError(
                "pytorch-forecasting and lightning are required. "
                "Install with: pip install pytorch-forecasting lightning"
            )

        cfg = self.model_cfg
        feat_cfg = self.config.get("features", {})

        max_encoder_length = cfg.get("max_encoder_length", 24)
        max_prediction_length = cfg.get("max_prediction_length", 2)

        static_cats = feat_cfg.get("static_categoricals", [])
        tv_known = feat_cfg.get("time_varying_known", [])
        tv_unknown = [self.target_col] + feat_cfg.get("time_varying_unknown", [])
        all_cont = self._get_all_continuous_cols(feat_cfg)

        # Ensure required columns exist; silently drop missing ones
        existing_cols = train_df.columns.tolist()
        static_cats = [c for c in static_cats if c in existing_cols]
        tv_known = [c for c in tv_known if c in existing_cols]
        tv_unknown = [c for c in tv_unknown if c in existing_cols]
        all_cont = [c for c in all_cont if c in existing_cols and c != self.target_col]

        # Encode categoricals as strings (required by pytorch-forecasting)
        df = train_df.copy()
        for col in static_cats + ["qtr_bucket"] if "qtr_bucket" in tv_known else static_cats:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Time index: integer months since minimum date
        df = df.sort_values([self.date_col] + self.grain_cols)
        min_date = df[self.date_col].min()
        df["time_idx"] = ((df[self.date_col].dt.year - min_date.year) * 12 +
                          (df[self.date_col].dt.month - min_date.month))

        # Ensure group identifier
        group_ids = self.grain_cols if self.grain_cols else ["_group"]
        if not self.grain_cols:
            df["_group"] = "all"

        # Filter to series with enough history
        min_len = max_encoder_length + max_prediction_length
        counts = df.groupby(group_ids)["time_idx"].count()
        valid_groups = counts[counts >= min_len].index
        if hasattr(valid_groups, 'tolist'):
            df = df[df.set_index(group_ids).index.isin(valid_groups)].copy() if len(group_ids) > 1 \
                else df[df[group_ids[0]].isin(valid_groups)].copy()

        # Validation set: last max_prediction_length time steps
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
            static_categoricals=static_cats,
            time_varying_known_categoricals=[c for c in tv_known if df[c].dtype == object] if tv_known else [],
            time_varying_known_reals=[c for c in tv_known if df[c].dtype != object] + ["time_idx"],
            time_varying_unknown_reals=tv_unknown + all_cont,
            target_normalizer=GroupNormalizer(groups=group_ids, transformation="softplus"),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )

        val_ds = TimeSeriesDataSet.from_dataset(
            self._training_ds,
            df,
            predict=True,
            stop_randomization=True,
        )

        batch_size = cfg.get("batch_size", 64)
        num_workers = cfg.get("num_workers", 0)

        train_loader = self._training_ds.to_dataloader(
            train=True, batch_size=batch_size, num_workers=num_workers
        )
        val_loader = val_ds.to_dataloader(
            train=False, batch_size=batch_size * 2, num_workers=num_workers
        )

        # Model
        self._model = TemporalFusionTransformer.from_dataset(
            self._training_ds,
            learning_rate=cfg.get("learning_rate", 0.003),
            hidden_size=cfg.get("hidden_size", 45),
            attention_head_size=cfg.get("attention_head_size", 4),
            dropout=cfg.get("dropout", 0.2286),
            hidden_continuous_size=cfg.get("hidden_continuous_size", 26),
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )

        callbacks = [
            EarlyStopping(
                monitor="val_loss",
                patience=cfg.get("early_stopping_patience", 10),
                mode="min",
            ),
            LearningRateMonitor(),
        ]

        accelerator = cfg.get("accelerator", "cpu")
        devices = cfg.get("devices", 1)

        trainer = pl.Trainer(
            max_epochs=cfg.get("max_epochs", 50),
            accelerator=accelerator,
            devices=devices,
            gradient_clip_val=cfg.get("gradient_clip_val", 0.0469),
            callbacks=callbacks,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )

        print(f"[tft] Training TFT: encoder={max_encoder_length}, horizon={max_prediction_length}")
        trainer.fit(self._model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        self._trainer = trainer
        self._min_date = min_date
        self._group_ids = group_ids
        self._df_fit = df
        self._is_fitted = True
        print("[tft] Training complete.")

    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """Generate forecasts from the most recent context in context_df."""
        self._check_fitted()

        from pytorch_forecasting import TimeSeriesDataSet

        df = context_df.copy()
        for col in self._training_ds.static_categoricals:
            if col in df.columns:
                df[col] = df[col].astype(str)

        df["time_idx"] = ((df[self.date_col].dt.year - self._min_date.year) * 12 +
                          (df[self.date_col].dt.month - self._min_date.month))

        if not self.grain_cols:
            df["_group"] = "all"

        predict_ds = TimeSeriesDataSet.from_dataset(
            self._training_ds, df, predict=True, stop_randomization=True
        )
        batch_size = self.model_cfg.get("batch_size", 64)
        loader = predict_ds.to_dataloader(train=False, batch_size=batch_size)

        raw_preds, idx = self._model.predict(loader, return_index=True, return_x=False)
        # raw_preds shape: (n_series, horizon) — median by default

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
        print(f"[tft] Backtest cutoffs: {[str(c.date()) for c in cutoffs]}")

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

    def get_feature_importance(self) -> pd.DataFrame | None:
        """Returns TFT variable attention weights as feature importances."""
        self._check_fitted()
        try:
            interp = self._model.interpret_output(
                self._model.predict(
                    self._training_ds.to_dataloader(train=False, batch_size=64),
                    return_attention=True,
                )
            )
            imp = interp.get("encoder_variables", pd.DataFrame())
            if not imp.empty:
                imp.columns = ["feature", "importance"]
            return imp
        except Exception:
            return None

    @staticmethod
    def _get_all_continuous_cols(feat_cfg: dict) -> list:
        cont = feat_cfg.get("continuous", {})
        cols = []
        for group_cols in cont.values():
            cols.extend(group_cols)
        return cols
