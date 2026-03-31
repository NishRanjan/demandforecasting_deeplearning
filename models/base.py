"""
models/base.py
──────────────
Abstract base class that every forecasting model must implement.

The output contract for predict() and backtest() is the critical
shared interface — all downstream evaluation code depends on it.

Output schema for predict():
    [date_col, *grain_cols, "forecast", "q10", "q90", "model_name"]

Output schema for backtest():
    [date_col, *grain_cols, "forecast", "q10", "q90", "model_name", "actual", "series_segment"]
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.

    Subclasses must implement: name, fit, predict, backtest.
    get_feature_importance is optional — return None if not supported.
    """

    def __init__(self, config: dict):
        self.config = config
        self.data_cfg = config["data"]
        self.backtest_cfg = config["backtest"]
        self.eval_cfg = config.get("evaluation", {})

        # Each subclass reads its own section: config["models"]["<name>"]
        self.model_cfg = config["models"].get(self.name, {})

        self._is_fitted = False

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier: 'chronos', 'tft', 'nhits', 'deepar'."""
        ...

    @abstractmethod
    def fit(self, train_df: pd.DataFrame) -> None:
        """
        Train or load the model using train_df.
        For pre-trained models (Chronos), this loads the pipeline weights.
        For trainable models (TFT/NHiTS/DeepAR), this runs the training loop.

        Sets self._is_fitted = True on completion.
        """
        ...

    @abstractmethod
    def predict(self, context_df: pd.DataFrame, horizon: int) -> pd.DataFrame:
        """
        Generate a forecast of length `horizon` from the context in context_df.

        Parameters
        ----------
        context_df : DataFrame containing all history visible at prediction time.
                     Must include date_col, grain_cols, target_col, and all features.
        horizon    : Number of future periods to forecast.

        Returns
        -------
        DataFrame with columns:
            [date_col, *grain_cols, "forecast", "q10", "q90", "model_name"]

        Notes
        -----
        - "forecast" is the point forecast (median / best estimate).
        - "q10" and "q90" are prediction interval bounds.
          For models that don't produce intervals, fill with NaN.
        - All grain combinations present in context_df must appear in output.
        """
        ...

    @abstractmethod
    def backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run rolling-origin backtest using settings from config['backtest'].

        Uses date_utils.get_backtest_cutoffs() to determine cutoff dates,
        then calls fit() + predict() for each window.

        Returns
        -------
        DataFrame with columns:
            [date_col, *grain_cols, "forecast", "q10", "q90",
             "model_name", "actual", "series_segment"]
        """
        ...

    def get_feature_importance(self) -> pd.DataFrame | None:
        """
        Returns a DataFrame with columns ["feature", "importance"] if supported.
        Returns None for models that don't support feature importance (e.g. Chronos).
        """
        return None

    # ── Convenience helpers available to all subclasses ──────────────────────

    @property
    def date_col(self) -> str:
        return self.data_cfg["date_col"]

    @property
    def target_col(self) -> str:
        return self.data_cfg["target_col"]

    @property
    def grain_cols(self) -> list:
        return self.data_cfg.get("grain_cols", [])

    def _check_fitted(self):
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.name}: predict() called before fit(). Call fit() first."
            )

    def _empty_forecast_df(self) -> pd.DataFrame:
        """Returns an empty DataFrame with the correct output schema."""
        cols = [self.date_col] + self.grain_cols + ["forecast", "q10", "q90", "model_name"]
        return pd.DataFrame(columns=cols)
