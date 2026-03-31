"""
tracking/mlflow_logger.py
──────────────────────────
Thin MLflow wrapper. All methods are no-ops when use_mlflow=False,
so model code never needs if/else guards for tracking.

Works with both:
- Databricks managed MLflow (auto-configured via environment)
- Local MLflow server (set MLFLOW_TRACKING_URI in .env)
"""

from __future__ import annotations

import os
from typing import Any


class MLflowLogger:

    def __init__(self, config: dict):
        platform_cfg = config.get("platform", {})
        self.enabled = platform_cfg.get("use_mlflow", False)
        self.experiment_name = platform_cfg.get("mlflow_experiment_name", "demand_forecast")
        self._run = None

        if self.enabled:
            try:
                import mlflow
                self._mlflow = mlflow
                mlflow.set_experiment(self.experiment_name)
                print(f"[mlflow] Tracking enabled: experiment='{self.experiment_name}'")
            except ImportError:
                print("[mlflow] WARNING: mlflow not installed. Disabling tracking.")
                self.enabled = False

    def start_run(self, run_name: str | None = None, tags: dict | None = None):
        if not self.enabled:
            return
        self._run = self._mlflow.start_run(run_name=run_name, tags=tags or {})

    def end_run(self):
        if not self.enabled or self._run is None:
            return
        self._mlflow.end_run()
        self._run = None

    def log_params(self, params: dict):
        if not self.enabled:
            return
        # MLflow limits param values to strings of ≤500 chars
        safe_params = {k: str(v)[:499] for k, v in params.items()}
        self._mlflow.log_params(safe_params)

    def log_metrics(self, metrics: dict, step: int | None = None):
        if not self.enabled:
            return
        # Filter out non-numeric and NaN values
        clean = {}
        for k, v in metrics.items():
            try:
                fv = float(v)
                if fv == fv:  # NaN check
                    clean[k] = fv
            except (TypeError, ValueError):
                pass
        if clean:
            self._mlflow.log_metrics(clean, step=step)

    def log_artifact(self, local_path: str, artifact_path: str | None = None):
        if not self.enabled:
            return
        if os.path.exists(local_path):
            self._mlflow.log_artifact(local_path, artifact_path)

    def log_model(self, model, artifact_path: str = "model"):
        if not self.enabled:
            return
        try:
            self._mlflow.pytorch.log_model(model, artifact_path)
        except Exception as e:
            print(f"[mlflow] Could not log model: {e}")

    def __enter__(self):
        self.start_run()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_run()
        return False
