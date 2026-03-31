"""
run_pipeline.py
───────────────
CLI entry point for the demand forecasting pipeline.

Usage:
    python run_pipeline.py --config config/forecast_config.yaml
    python run_pipeline.py --config config/forecast_config.yaml --models chronos tft
    python run_pipeline.py --config config/forecast_config.yaml --skip-backtest

The pipeline also exposes main() for calling programmatically
(e.g. from a Databricks notebook cell or another script).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import yaml


# ── Ensure the demand_forecasting directory is on the Python path ─────────────
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))


# ── Model registry ────────────────────────────────────────────────────────────
# To add a new model: create models/<name>_model.py with a class that subclasses
# BaseForecaster, then add one entry here.
MODEL_REGISTRY = {
    "chronos": ("models.chronos_model", "ChronosForecaster"),
    "tft":     ("models.tft_model",     "TFTForecaster"),
    "nhits":   ("models.nhits_model",   "NHiTSForecaster"),
    "deepar":  ("models.deepar_model",  "DeepARForecaster"),
}


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_model(model_name: str, config: dict):
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(MODEL_REGISTRY)}"
        )
    module_path, class_name = MODEL_REGISTRY[model_name]
    import importlib
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls(config)


def main(config_path: str, models_filter: list[str] | None = None, skip_backtest: bool = False):
    """
    Run the full pipeline:
      1. Load config & data
      2. Preprocess & feature engineering
      3. Backtest + forward forecast for each enabled model
      4. Compare models and save results
    """
    from data.loader import load_data
    from data.preprocessor import preprocess
    from data.feature_engineering import engineer_features
    from evaluation.comparison import compare_models, print_comparison
    from tracking.mlflow_logger import MLflowLogger
    from utils.platform import display_df, get_platform_mode

    print("\n" + "=" * 60)
    print("  DEMAND FORECASTING PIPELINE")
    print("=" * 60)

    # ── 1. Config ──────────────────────────────────────────────────────────────
    config = load_config(config_path)
    mode = get_platform_mode(config)
    print(f"[pipeline] Platform mode: {mode}")

    # ── 2. Data ────────────────────────────────────────────────────────────────
    raw_df = load_data(config)
    clean_df = preprocess(raw_df, config)
    featured_df = engineer_features(clean_df, config)

    # ── 3. Determine which models to run ──────────────────────────────────────
    enabled_models = [
        name for name, cfg in config.get("models", {}).items()
        if cfg.get("enabled", False)
    ]
    if models_filter:
        enabled_models = [m for m in enabled_models if m in models_filter]

    if not enabled_models:
        print("[pipeline] No models enabled. Check config['models'][*]['enabled'].")
        return

    print(f"[pipeline] Models to run: {enabled_models}")

    # ── 4. MLflow ──────────────────────────────────────────────────────────────
    logger = MLflowLogger(config)

    backtest_results: dict[str, pd.DataFrame] = {}
    forecast_results: dict[str, pd.DataFrame] = {}

    for model_name in enabled_models:
        print(f"\n{'─' * 60}")
        print(f"  Running: {model_name.upper()}")
        print(f"{'─' * 60}")

        model = get_model(model_name, config)

        logger.start_run(run_name=model_name, tags={"model": model_name})
        logger.log_params(config.get("models", {}).get(model_name, {}))
        logger.log_params({
            "encoder_length": config.get("models", {}).get(model_name, {}).get("max_encoder_length", "N/A"),
            "n_backtest_windows": config["backtest"]["n_rolling_windows"],
            "forecast_horizon": config["backtest"]["forecast_horizon"],
        })

        # Backtest
        if not skip_backtest:
            print(f"[{model_name}] Starting backtest...")
            bt_df = model.backtest(featured_df)
            backtest_results[model_name] = bt_df

            # Log aggregate backtest metrics
            valid = bt_df.dropna(subset=["actual", "forecast"])
            if not valid.empty:
                from evaluation.metrics import wmape, r_squared, bias
                bt_wmape = wmape(valid["actual"].values, valid["forecast"].values)
                bt_r2 = r_squared(valid["actual"].values, valid["forecast"].values)
                bt_bias = bias(valid["actual"].values, valid["forecast"].values)
                logger.log_metrics({
                    "backtest_wmape": bt_wmape,
                    "backtest_r2": bt_r2,
                    "backtest_bias": bt_bias,
                })
                print(f"[{model_name}] Backtest WMAPE={bt_wmape:.4f}  R²={bt_r2:.4f}  Bias={bt_bias:.4f}")

        # Forward forecast (fit on all data)
        print(f"[{model_name}] Fitting on full dataset for forward forecast...")
        model.fit(featured_df)
        forward_horizon = config["backtest"]["forward_forecast_months"]
        fc_df = model.predict(featured_df, horizon=forward_horizon)
        forecast_results[model_name] = fc_df

        # Feature importance
        fi = model.get_feature_importance()
        if fi is not None and not fi.empty:
            fi_path = _ensure_output_dir(config) / f"{model_name}_feature_importance.csv"
            fi.to_csv(fi_path, index=False)
            logger.log_artifact(str(fi_path))

        logger.end_run()

    # ── 5. Save forward forecasts ──────────────────────────────────────────────
    fc_dir = Path(config["evaluation"].get("forecast_output_dir", "outputs/forecasts"))
    fc_dir.mkdir(parents=True, exist_ok=True)

    for model_name, fc_df in forecast_results.items():
        fc_path = fc_dir / f"{model_name}_forecast.csv"
        fc_df.to_csv(fc_path, index=False)
        print(f"[pipeline] Forecast saved: {fc_path}")

    # ── 6. Compare models ──────────────────────────────────────────────────────
    if backtest_results:
        grain_cols = config["data"].get("grain_cols", [])
        summary_df, per_series_df = compare_models(backtest_results, grain_cols=grain_cols)

        output_path = Path(config["evaluation"].get("output_path", "outputs/comparison_results.csv"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        summary_df.to_csv(output_path, index=False)

        per_series_path = output_path.parent / "per_series_metrics.csv"
        per_series_df.to_csv(per_series_path, index=False)

        print_comparison(summary_df)
        display_df(summary_df)

        print(f"[pipeline] Comparison saved: {output_path}")
        print(f"[pipeline] Per-series metrics saved: {per_series_path}")

        return summary_df, per_series_df, forecast_results

    return None, None, forecast_results


def _ensure_output_dir(config: dict) -> Path:
    out = Path(config["evaluation"].get("output_path", "outputs/comparison_results.csv")).parent
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── CLI ────────────────────────────────────────────────────────────────────────

def _parse_args():
    parser = argparse.ArgumentParser(
        description="Demand Forecasting Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --config config/forecast_config.yaml
  python run_pipeline.py --config config/forecast_config.yaml --models chronos tft
  python run_pipeline.py --config config/forecast_config.yaml --skip-backtest
        """
    )
    parser.add_argument(
        "--config", "-c",
        default="config/forecast_config.yaml",
        help="Path to YAML config file (default: config/forecast_config.yaml)",
    )
    parser.add_argument(
        "--models", "-m",
        nargs="+",
        default=None,
        help="Subset of models to run (e.g. --models chronos tft). "
             "Runs all enabled models in config if omitted.",
    )
    parser.add_argument(
        "--skip-backtest",
        action="store_true",
        default=False,
        help="Skip backtest; run forward forecast only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(
        config_path=args.config,
        models_filter=args.models,
        skip_backtest=args.skip_backtest,
    )
